"""Pseudo-passage generator: round-1 generation (via QueryGym) and
round-2+ history-aware query expansion.

Round 1 delegates to QueryGym's generation implementation for
reproducibility.  Round 2+ LLM calls go through ``LLMClient`` so they
are cached and retried.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

import querygym as qg

from ..llm_client import LLMClient
from ..query_expansion import build_expanded_query, build_passage_count_query
from ..schemas import Round

logger = logging.getLogger(__name__)


# ----- Pydantic schemas for structured output -------------------------------


class PseudoPassages(BaseModel):
    """Pydantic schema for `chat.completions.parse` generator output."""

    passages: list[str] = Field(default_factory=list)


# ----- History-aware iterative prompt (built-in fallback; prefer YAML) ------

_ITER_SYSTEM = (
    "You are a search reformulation expert. You write natural-English pseudo-"
    "passages used to expand dense retrieval queries."
)

_ITER_USER = """You are improving search coverage for an information retrieval task. Your job is to generate a pseudo-passage that, when used as a dense retrieval query, retrieves documents NOT yet seen and better addresses aspects of the query underserved by previous retrievals.

ORIGINAL QUERY: {q0}

PREVIOUS REFORMULATIONS (round 1 to round {i_minus_1}):
{previous_reformulations_block}

DOCUMENTS RETRIEVED AND ASSESSED SO FAR (deduplicated):

  Highly relevant (score 3) -- already covered, do NOT duplicate:
{score3_block}

  Relevant (score 2):
{score2_block}

  Marginally relevant (score 1) -- these matched but missed the intent:
{score1_block}

  Irrelevant (score 0) -- retrieval false positives, suppress these patterns:
{score0_block}

YOUR TASK:
Generate one NEW pseudo-passage that:
  1. Covers an aspect of the query that the score-3 documents above do NOT address.
  2. Uses vocabulary and phrasings DIFFERENT from previous reformulations (otherwise retrieval will return the same documents).
  3. Avoids the topical patterns of the score-0 (irrelevant) documents -- those are retrieval false positives we want to suppress.
  4. Is concise, informative, and natural English (like a real document passage, 3-5 sentences).

Return ONLY the pseudo-passage text, nothing else.
"""


def load_iter_prompt_yaml(path: str | Path | None) -> tuple[str, str] | None:
    """Load (system, user_template) from a YAML file; None if path missing or invalid."""
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        logger.warning("iter_prompt_yaml not found (%s); using built-in iter prompts", p)
        return None
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        logger.warning("iter_prompt_yaml %s is not a mapping; using built-in", p)
        return None
    system = data.get("system")
    user = data.get("user")
    if not isinstance(system, str) or not isinstance(user, str):
        logger.warning("iter_prompt_yaml %s missing system/user strings; using built-in", p)
        return None
    return system.strip(), user.strip()


# Default per-score evidence snippet limits when generator.max_chars is null.
_DEFAULT_EVIDENCE_MAX_CHARS: dict[int, int] = {3: 1200, 2: 800, 1: 400, 0: 400}


# ---------------------------------------------------------------------------


class PseudoPassageGenerator:
    """Round-1 generation (via QueryGym) + round-2+ history-aware generator."""

    def __init__(
        self,
        llm: LLMClient,
        model: str,
        *,
        k_pseudo_refs: int = 5,
        alpha: int = 5,
        temperature: float = 1.0,
        max_tokens: int = 1024,
        round1_temperature: float = 1.0,
        round1_max_tokens: int = 128,
        num_threads: int = 4,
        iter_prompt_yaml: str | Path | None = None,
        iter_mode: str = "llm",
        relevance_feedback_max_docs: int = 5,
        relevance_feedback_min_score: int = 2,
        rf_fallback_prompt_yaml: str | Path | None = None,
        evidence_max_chars: int | None = None,
        skip_round1_generation: bool = False,
        accumulate_pseudo_refs_from_r2: bool = False,
        log_prompts: bool = False,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        self.llm = llm
        self.model = model
        self.k = k_pseudo_refs
        self.log_prompts = log_prompts
        self.evidence_max_chars = evidence_max_chars
        self.alpha = alpha
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.num_threads = max(1, num_threads)
        self.iter_mode = iter_mode
        self.rf_max_docs = relevance_feedback_max_docs
        self.rf_min_score = relevance_feedback_min_score
        self.skip_round1_generation = skip_round1_generation
        self.accumulate_pseudo_refs_from_r2 = accumulate_pseudo_refs_from_r2
        loaded = load_iter_prompt_yaml(iter_prompt_yaml)
        if loaded is None:
            self._iter_system, self._iter_user_template = _ITER_SYSTEM, _ITER_USER
        else:
            self._iter_system, self._iter_user_template = loaded

        rf_fallback = load_iter_prompt_yaml(rf_fallback_prompt_yaml)
        if rf_fallback is not None:
            self._rf_fallback_system, self._rf_fallback_user = rf_fallback
            logger.info("RF fallback prompt loaded from %s", rf_fallback_prompt_yaml)
        else:
            self._rf_fallback_system = self._iter_system
            self._rf_fallback_user = self._iter_user_template
            if iter_mode == "relevance_feedback" and rf_fallback_prompt_yaml:
                logger.warning(
                    "RF fallback prompt not found (%s); using iter_prompt_yaml instead",
                    rf_fallback_prompt_yaml,
                )

        if skip_round1_generation:
            self._qg_generator = None
            logger.info(
                "Round-1 generation SKIPPED (skip_round1_generation=True); "
                "R1 will use original query only."
            )
        else:
            llm_config: dict[str, Any] = {
                "temperature": round1_temperature,
                "max_tokens": round1_max_tokens,
            }
            if base_url:
                llm_config["base_url"] = base_url
            if api_key:
                llm_config["api_key"] = api_key

            self._qg_generator = qg.create_reformulator(
                "mugi",
                model=model,
                params={
                    "num_docs": k_pseudo_refs,
                    "adaptive_times": alpha,
                    "temperature": round1_temperature,
                    "max_tokens": round1_max_tokens,
                    "parallel": num_threads > 1,
                },
                llm_config=llm_config,
            )
            logger.info(
                "Round-1 pseudo-passage generation delegated to QueryGym "
                "(model=%s, k=%d, alpha=%d, temperature=%.1f, max_tokens=%d)",
                model, k_pseudo_refs, alpha, round1_temperature, round1_max_tokens,
            )

    # ------------------------------------------------------------- Round 1

    def generate_round1(self, q0: str) -> tuple[str, list[str], dict[str, Any]]:
        """Returns (expanded_query, pseudo_refs, meta).

        When ``skip_round1_generation`` is True, returns the original query as-is
        (no LLM expansion). Otherwise delegates to QueryGym for pseudo-passage
        generation followed by alpha-repetition expansion.
        """
        if self.skip_round1_generation:
            meta = {
                "mode": "round1_original_query",
                "k": 0,
                "alpha": self.alpha,
                "expansion": {
                    "alpha": self.alpha,
                    "query_chars": len(q0),
                    "docs_chars": 0,
                    "repetition_times": 1,
                    "num_pseudo_refs": 0,
                },
                "usage": {
                    "llm_calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "estimated_usd": 0.0,
                },
            }
            return q0.strip(), [], meta

        query_item = qg.QueryItem(qid="round1", text=q0)
        result = self._qg_generator.reformulate(query_item)

        pseudo_refs: list[str] = result.metadata.get("pseudo_docs", [])
        expanded = result.reformulated

        meta = {
            "mode": "round1_generation",
            "k": self.k,
            "alpha": self.alpha,
            "expansion": {
                "alpha": result.metadata.get("adaptive_times", self.alpha),
                "query_chars": result.metadata.get("query_len", len(q0)),
                "docs_chars": result.metadata.get("docs_len", 0),
                "repetition_times": result.metadata.get("repetition_times", 0),
                "num_pseudo_refs": result.metadata.get("num_docs", len(pseudo_refs)),
            },
            "usage": {
                "llm_calls": len(pseudo_refs),
                "input_tokens": 0,
                "output_tokens": 0,
                "estimated_usd": 0.0,
            },
        }
        return expanded, pseudo_refs, meta

    # ----------------------------------------------------------- Round 2+

    @staticmethod
    def _format_doc_block(scored_docs: list[tuple[str, str]], max_chars: int = 800) -> str:
        if not scored_docs:
            return "    (none)\n"
        lines = []
        for doc_id, snippet in scored_docs:
            s = (snippet or "").replace("\n", " ").strip()
            if len(s) > max_chars:
                s = s[: max_chars - 1] + "..."
            lines.append(f'    [doc_id {doc_id}] "{s}"')
        return "\n".join(lines) + "\n"

    def _evidence_block_max_chars(self, score: int) -> int:
        if self.evidence_max_chars is not None:
            return self.evidence_max_chars
        return _DEFAULT_EVIDENCE_MAX_CHARS.get(score, 800)

    @staticmethod
    def _format_previous_reformulations(
        rounds: list[Round], preview_chars: int = 200
    ) -> str:
        if not rounds:
            return "  (none)\n"
        lines = []
        for r in rounds:
            lines.append(f"  Round {r.round_idx} pseudo-passages:")
            for i, p in enumerate(r.pseudo_refs, start=1):
                preview = (p or "").replace("\n", " ").strip()
                if len(preview) > preview_chars:
                    preview = preview[: preview_chars - 1] + "..."
                lines.append(f'    P{r.round_idx}.{i}: "{preview}"')
        return "\n".join(lines) + "\n"

    def _iter_one(
        self,
        user_prompt: str,
        idx: int,
        round_num: int,
        system_prompt: str | None = None,
    ) -> tuple[str, int, int, float]:
        """Generate a single iterative pseudo-passage. Returns (text, in_tok, out_tok, usd)."""
        messages = [
            {"role": "system", "content": system_prompt or self._iter_system},
            {"role": "user", "content": user_prompt},
        ]
        res = self.llm.chat_text(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            cache_namespace=f"iter_r{round_num}_idx{idx}",
        )
        text = (res.text or "").strip()
        if text.startswith("{"):
            import json as _json
            try:
                obj = _json.loads(text)
                if isinstance(obj, dict) and isinstance(obj.get("passages"), list) and obj["passages"]:
                    text = str(obj["passages"][0]).strip()
            except Exception:
                pass
        text = text.strip('"').strip("'")
        logger.debug(
            "[iter-response round=%d idx=%d] %s",
            round_num, idx, text[:300] + ("..." if len(text) > 300 else ""),
        )
        if self.log_prompts:
            preview = text[:300] + ("..." if len(text) > 300 else "")
            print(f"[ITER RESPONSE round={round_num} idx={idx}] {preview}", flush=True)
        return text, res.input_tokens, res.output_tokens, res.usd

    def _collect_relevance_feedback(
        self,
        doc_snippets: dict[str, str],
        doc_judge_scores: dict[str, int],
        doc_best_score: dict[str, float] | None = None,
    ) -> list[str]:
        """Collect up to rf_max_docs real passages with score >= rf_min_score.

        Prioritises score-3 over score-2; within each tier picks highest retrieval
        score (best observed score in ``doc_best_score``) when available.
        """
        by_score: dict[int, list[str]] = {}
        for doc_id, score in doc_judge_scores.items():
            if score >= self.rf_min_score:
                by_score.setdefault(score, []).append(doc_id)

        def _rank_key(doc_id: str) -> tuple[float, str]:
            if doc_best_score is not None:
                return (-doc_best_score.get(doc_id, float("-inf")), doc_id)
            return (0.0, doc_id)

        passages: list[str] = []
        for score_tier in (3, 2):
            candidates = sorted(by_score.get(score_tier, []), key=_rank_key)
            for doc_id in candidates:
                if len(passages) >= self.rf_max_docs:
                    break
                snippet = doc_snippets.get(doc_id, "").strip()
                if snippet:
                    passages.append(snippet)
            if len(passages) >= self.rf_max_docs:
                break
        return passages

    @staticmethod
    def _build_append_query(
        q0: str, passages: list[str]
    ) -> tuple[str, dict[str, Any]]:
        """Expand query: repeat q0 once per appended passage, then append all text."""
        expanded, expansion_meta = build_passage_count_query(q0, passages)
        return expanded, expansion_meta

    @staticmethod
    def _collect_accumulated_pseudo_refs(
        rounds_so_far: list[Round], current_passages: list[str]
    ) -> list[str]:
        """Collect pseudo-refs from rounds >= 2 plus the current round (skip R1)."""
        accumulated: list[str] = []
        for rnd in rounds_so_far:
            if rnd.round_idx >= 2:
                accumulated.extend(rnd.pseudo_refs)
        accumulated.extend(current_passages)
        return accumulated

    def _generate_iter_passages(
        self,
        q0: str,
        rounds_so_far: list[Round],
        doc_snippets: dict[str, str],
        doc_judge_scores: dict[str, int],
        *,
        use_fallback_prompt: bool = False,
    ) -> tuple[list[str], dict[str, Any]]:
        """Generate k pseudo-passages via the iter (or fallback) LLM prompt."""
        by_score: dict[int, list[tuple[str, str]]] = {0: [], 1: [], 2: [], 3: []}
        for doc_id, score in doc_judge_scores.items():
            snippet = doc_snippets.get(doc_id, "")
            by_score.setdefault(score, []).append((doc_id, snippet))

        previous_block = self._format_previous_reformulations(rounds_so_far)
        round_num = len(rounds_so_far) + 1

        system_prompt = self._rf_fallback_system if use_fallback_prompt else self._iter_system
        user_template = self._rf_fallback_user if use_fallback_prompt else self._iter_user_template

        user_prompt = user_template.format(
            q0=q0,
            i_minus_1=len(rounds_so_far),
            previous_reformulations_block=previous_block,
            score3_block=self._format_doc_block(
                by_score.get(3, []), max_chars=self._evidence_block_max_chars(3)
            ),
            score2_block=self._format_doc_block(
                by_score.get(2, []), max_chars=self._evidence_block_max_chars(2)
            ),
            score1_block=self._format_doc_block(
                by_score.get(1, []), max_chars=self._evidence_block_max_chars(1)
            ),
            score0_block=self._format_doc_block(
                by_score.get(0, []), max_chars=self._evidence_block_max_chars(0)
            ),
            k=self.k,
        )

        logger.debug(
            "[iter-prompt round=%d]\n--- SYSTEM ---\n%s\n--- USER ---\n%s\n--- END ---",
            round_num, system_prompt, user_prompt,
        )
        if self.log_prompts:
            sep = "=" * 72
            print(f"\n{sep}\n[ITER PROMPT  round={round_num}]\n{sep}")
            print(f"--- SYSTEM ---\n{system_prompt}")
            print(f"--- USER ---\n{user_prompt}")
            print(sep, flush=True)

        usage_total: dict[str, Any] = {
            "llm_calls": 0, "input_tokens": 0, "output_tokens": 0, "estimated_usd": 0.0,
        }
        results: list[tuple[int, str]] = []

        with ThreadPoolExecutor(max_workers=min(self.num_threads, self.k)) as ex:
            futures = {
                ex.submit(self._iter_one, user_prompt, i, round_num, system_prompt): i
                for i in range(self.k)
            }
            for fut in as_completed(futures):
                i = futures[fut]
                text, in_tok, out_tok, usd = fut.result()
                results.append((i, text))
                usage_total["llm_calls"] += 1
                usage_total["input_tokens"] += in_tok
                usage_total["output_tokens"] += out_tok
                usage_total["estimated_usd"] += usd

        results.sort(key=lambda t: t[0])
        passages = [t for _, t in results if t]
        usage_total["n_llm_passages"] = len(passages)
        if use_fallback_prompt:
            usage_total["prompt"] = "pseudo_passage_generator.yaml"
        return passages, usage_total

    def _generate_iter_llm(
        self,
        q0: str,
        rounds_so_far: list[Round],
        doc_snippets: dict[str, str],
        doc_judge_scores: dict[str, int],
        use_fallback_prompt: bool = False,
    ) -> tuple[str, list[str], dict[str, Any]]:
        """LLM-based iterative generation with alpha-repetition expansion."""
        passages, usage_total = self._generate_iter_passages(
            q0, rounds_so_far, doc_snippets, doc_judge_scores,
            use_fallback_prompt=use_fallback_prompt,
        )

        round_num = len(rounds_so_far) + 1
        if self.accumulate_pseudo_refs_from_r2 and round_num >= 3:
            retrieval_passages = self._collect_accumulated_pseudo_refs(
                rounds_so_far, passages
            )
            expanded, expansion_meta = build_expanded_query(
                q0, retrieval_passages, alpha=self.alpha
            )
            mode_label = "iter_accumulated_from_r2"
        else:
            retrieval_passages = passages
            expanded, expansion_meta = build_expanded_query(q0, passages, alpha=self.alpha)
            mode_label = "iter_rf_fallback" if use_fallback_prompt else "iter_history_aware"

        meta = {
            "mode": mode_label,
            "k": self.k,
            "alpha": self.alpha,
            "expansion": expansion_meta,
            "usage": usage_total,
            "n_passages_returned": len(passages),
            "n_retrieval_passages": len(retrieval_passages),
        }
        if use_fallback_prompt:
            meta["prompt"] = "pseudo_passage_generator.yaml"
        return expanded, passages, meta

    def _generate_iter_combined(
        self,
        q0: str,
        rounds_so_far: list[Round],
        doc_snippets: dict[str, str],
        doc_judge_scores: dict[str, int],
        doc_best_score: dict[str, float] | None = None,
    ) -> tuple[str, list[str], dict[str, Any]]:
        """LLM pseudo-passages + top relevance-feedback real passages by retrieval score."""
        llm_passages, usage_total = self._generate_iter_passages(
            q0, rounds_so_far, doc_snippets, doc_judge_scores,
        )
        rf_passages = self._collect_relevance_feedback(
            doc_snippets, doc_judge_scores, doc_best_score,
        )
        all_passages = llm_passages + rf_passages

        expanded, expansion_meta = self._build_append_query(q0, all_passages)
        meta = {
            "mode": "iter_combined",
            "k": self.k,
            "n_llm_passages": len(llm_passages),
            "n_rf_appended": len(rf_passages),
            "rf_max_docs": self.rf_max_docs,
            "rf_min_score": self.rf_min_score,
            "expansion": expansion_meta,
            "usage": usage_total,
            "n_passages_returned": len(all_passages),
            "prompt": "pseudo_passage_generator.yaml",
        }
        logger.info(
            "Combined iter (round %d): %d LLM pseudo + %d RF real "
            "(max=%d, min_score=%d, repetition_times=%d)",
            len(rounds_so_far) + 1,
            len(llm_passages),
            len(rf_passages),
            self.rf_max_docs,
            self.rf_min_score,
            expansion_meta["repetition_times"],
        )
        return expanded, all_passages, meta

    def generate_iter(
        self,
        q0: str,
        rounds_so_far: list[Round],
        doc_snippets: dict[str, str],
        doc_judge_scores: dict[str, int],
        doc_best_score: dict[str, float] | None = None,
    ) -> tuple[str, list[str], dict[str, Any]]:
        """Round 2+: dispatches by ``iter_mode``.

        - ``llm``: history-aware pseudo-passages + alpha-repetition expansion.
        - ``relevance_feedback``: append real score>=rf_min_score passages only.
        - ``combined``: iter YAML pseudo-passages + up to rf_max_docs real
          passages (score>=rf_min_score, highest retrieval score first);
          ``repetition_times`` = total appended doc count.
        """
        if self.iter_mode == "combined":
            return self._generate_iter_combined(
                q0, rounds_so_far, doc_snippets, doc_judge_scores, doc_best_score,
            )

        if self.iter_mode == "relevance_feedback":
            rf_passages = self._collect_relevance_feedback(
                doc_snippets, doc_judge_scores, doc_best_score,
            )
            if rf_passages:
                expanded, expansion_meta = self._build_append_query(q0, rf_passages)
                n_docs = len(rf_passages)
                n_score3 = sum(1 for s in doc_judge_scores.values() if s == 3)
                meta = {
                    "mode": "iter_relevance_feedback",
                    "k": n_docs,
                    "expansion": expansion_meta,
                    "usage": {"llm_calls": 0, "input_tokens": 0, "output_tokens": 0, "estimated_usd": 0.0},
                    "n_passages_returned": n_docs,
                    "rf_source_scores": [3] * min(n_docs, n_score3)
                        + [2] * (n_docs - min(n_docs, n_score3)),
                }
                logger.info(
                    "Relevance feedback: using %d real passages (round %d)",
                    len(rf_passages), len(rounds_so_far) + 1,
                )
                return expanded, rf_passages, meta
            logger.info(
                "No score>=%d docs yet (round %d); falling back to LLM generation",
                self.rf_min_score, len(rounds_so_far) + 1,
            )
            return self._generate_iter_llm(
                q0, rounds_so_far, doc_snippets, doc_judge_scores,
                use_fallback_prompt=True,
            )

        return self._generate_iter_llm(q0, rounds_so_far, doc_snippets, doc_judge_scores)

    # Backward-compatible alias used by SEEKRunner
    reformulate_round1 = generate_round1
    reformulate_iter = generate_iter
