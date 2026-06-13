"""SEEK runner: iterative retrieval loop — generate, retrieve, assess, repeat.

The control flow is deterministic (no LLM "decides what to do next"):

  * ``SEEKRunner`` owns per-query state (``SearchHistory``) and exposes
    a single public entry point ``run(query_id, query) -> SearchHistory``.
  * Each round calls ``PseudoPassageGenerator -> DenseRetriever -> Judge``,
    deduplicates by doc_id, updates the history, and checks termination.
  * State and per-round metadata are persisted to a JSON trace file.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from .schemas import Document, JudgeResult, Round, SearchHistory
from .tools.assessor import Judge
from .tools.generator import PseudoPassageGenerator
from .tools.retriever import DenseRetriever

logger = logging.getLogger(__name__)

# Must match the cap used in Round.to_dict() for retrieved_doc_ids / new_doc_ids.
TRACE_TOP_K = 10


def _format_dense_query(original_query: str, pseudo_refs: list[str]) -> str:
    """Format a structured query string for dense retrieval.

    Produces a layout that accurately describes the compound input to the
    encoder:

        Post: <original question>

        Relevant pseudo-passages:
        1. <passage 1>
        2. <passage 2>
        ...

    This is used when ``query_format: "structured"`` is set in config.
    """
    parts = [f"Post: {original_query.strip()}"]
    if pseudo_refs:
        parts.append("\nRelevant pseudo-passages:")
        for i, p in enumerate(pseudo_refs, start=1):
            parts.append(f"{i}. {p.strip()}")
    return "\n".join(parts)


class SEEKRunner:
    """Single-query orchestrator. Stateless across queries; spawn one per query."""

    def __init__(
        self,
        retriever: DenseRetriever,
        judge: Judge,
        generator: PseudoPassageGenerator,
        *,
        max_rounds: int = 5,
        judge_depth: int = 10,
        termination_all_score_3: bool = True,
        quality_saturation_min_score: int = 2,
        termination_coverage_saturation: bool = True,
        saturation_min_new_docs: int = 3,
        saturation_consecutive_rounds: int = 2,
        log_per_round: bool = True,
        evidence_scope: str = "all",
        query_format: str = "expanded",
    ):
        self.retriever = retriever
        self.judge = judge
        self.generator = generator
        self.max_rounds = max_rounds
        self.judge_depth = judge_depth
        self.termination_all_score_3 = termination_all_score_3
        self.quality_saturation_min_score = quality_saturation_min_score
        self.termination_coverage_saturation = termination_coverage_saturation
        self.saturation_min_new_docs = saturation_min_new_docs
        self.saturation_consecutive_rounds = saturation_consecutive_rounds
        self.log_per_round = log_per_round
        self.query_format = query_format
        self.evidence_scope = evidence_scope

    # ----------------------------------------------------------- helpers

    def _accumulate_usage(self, hist: SearchHistory, usage: dict[str, Any]) -> None:
        for k, v in usage.items():
            if k in hist.usage:
                hist.usage[k] = (hist.usage[k] or 0) + (v or 0)

    def _filter_new(self, docs: list[Document], seen: set[str]) -> tuple[list[Document], list[Document]]:
        new, dupes = [], []
        for d in docs:
            (dupes if d.doc_id in seen else new).append(d)
        return new, dupes

    def _termination_after_round(
        self,
        n_new: int,
        prev_low_new_streak: int,
    ) -> tuple[str | None, int]:
        """Return (reason, updated_low_new_streak) for coverage saturation only."""
        if self.termination_coverage_saturation:
            if n_new < self.saturation_min_new_docs:
                streak = prev_low_new_streak + 1
            else:
                streak = 0
            if streak >= self.saturation_consecutive_rounds:
                return "coverage_saturation", streak
            return None, streak
        return None, prev_low_new_streak

    # -------------------------------------------------------------- main

    def run(self, query_id: str, query: str) -> SearchHistory:
        hist = SearchHistory(query_id=query_id, original_query=query)
        t0 = time.time()
        low_new_streak = 0

        for round_idx in range(1, self.max_rounds + 1):
            rd = Round(round_idx=round_idx)

            # ---- Generator ---------------------------------------------------
            if round_idx == 1:
                expanded, refs, meta = self.generator.generate_round1(query)
            else:
                if self.evidence_scope == "previous_round" and hist.rounds:
                    prev_round = hist.rounds[-1]
                    prev_doc_ids = set(prev_round.retrieved_doc_ids)
                    scope_snippets = {
                        did: s for did, s in hist.doc_snippets.items()
                        if did in prev_doc_ids
                    }
                    scope_judge_scores = {
                        did: s for did, s in hist.doc_judge_scores.items()
                        if did in prev_doc_ids
                    }
                elif self.evidence_scope == "previous_round_new" and hist.rounds:
                    prev_round = hist.rounds[-1]
                    prev_new_ids = set(prev_round.new_doc_ids)
                    scope_snippets = {
                        did: s for did, s in hist.doc_snippets.items()
                        if did in prev_new_ids
                    }
                    scope_judge_scores = {
                        did: s for did, s in hist.doc_judge_scores.items()
                        if did in prev_new_ids
                    }
                else:
                    scope_snippets = hist.doc_snippets
                    scope_judge_scores = hist.doc_judge_scores

                expanded, refs, meta = self.generator.generate_iter(
                    q0=query,
                    rounds_so_far=hist.rounds,
                    doc_snippets=scope_snippets,
                    doc_judge_scores=scope_judge_scores,
                    doc_best_score=hist.doc_best_score,
                )

            rd.pseudo_refs = refs
            rd.expanded_query = expanded
            rd.generator_meta = {k: v for k, v in meta.items() if k != "usage"}
            self._accumulate_usage(hist, meta.get("usage", {}))

            # ---- Retriever ---------------------------------------------------
            if self.query_format == "structured":
                search_query = _format_dense_query(query, refs)
            else:
                search_query = expanded
            try:
                docs = self.retriever.search(search_query)
            except Exception as e:
                logger.exception("Retriever error in round %d for %s: %s", round_idx, query_id, e)
                docs = []
            hist.usage["retrieval_calls"] = (hist.usage.get("retrieval_calls", 0) or 0) + 1
            rd.retrieved_doc_ids = [d.doc_id for d in docs]
            for d in docs:
                # Track best retrieval score across rounds for tiebreaks.
                prev = hist.doc_best_score.get(d.doc_id, float("-inf"))
                if d.retrieval_score > prev:
                    hist.doc_best_score[d.doc_id] = float(d.retrieval_score)
                rd.retrieval_scores[d.doc_id] = float(d.retrieval_score)

            # ---- Dedup -------------------------------------------------------
            new_docs, dup_docs = self._filter_new(docs, hist.seen_doc_ids)
            rd.n_new_docs_pool = len(new_docs)
            rd.duplicate_count = len(dup_docs)

            top_retrieved = docs[:TRACE_TOP_K]
            new_in_top, _ = self._filter_new(top_retrieved, hist.seen_doc_ids)
            rd.new_doc_ids = [d.doc_id for d in new_in_top]

            # Assess only new docs in the trace top-K (same set as new_doc_ids).
            new_in_top_sorted = sorted(
                new_in_top, key=lambda d: d.retrieval_score, reverse=True
            )
            docs_to_judge = new_in_top_sorted[: self.judge_depth]

            # ---- Assessor ----------------------------------------------------
            judge_results: list[JudgeResult] = []
            judge_usage: dict[str, Any] = {}
            if docs_to_judge:
                try:
                    judge_results, judge_usage = self.judge.judge_batch(query, docs_to_judge)
                except Exception as e:
                    logger.exception("Assessor error in round %d for %s: %s", round_idx, query_id, e)
                    judge_results = [
                        JudgeResult(
                            doc_id=d.doc_id,
                            score=None,
                            raw_output=f"ERROR: {e}",
                            model=self.judge.model,
                            parse_failed=True,
                        )
                        for d in docs_to_judge
                    ]
            self._accumulate_usage(hist, judge_usage)

            # ---- Persist history -------------------------------------------
            for d, jr in zip(docs_to_judge, judge_results):
                rd.judge_results[d.doc_id] = jr
                hist.seen_doc_ids.add(d.doc_id)
                hist.doc_snippets[d.doc_id] = d.content
                effective_score = jr.score if jr.score is not None else 0
                prev = hist.doc_judge_scores.get(d.doc_id, -1)
                if effective_score > prev:
                    hist.doc_judge_scores[d.doc_id] = effective_score
                if jr.parse_failed:
                    hist.doc_judge_parse_failed.add(d.doc_id)
            hist.rounds.append(rd)

            if self.log_per_round:
                self._log_round(query_id, rd)

            # ---- Quality saturation check ----------------------------------
            if self.termination_all_score_3 and top_retrieved:
                unjudged = [d for d in top_retrieved
                            if d.doc_id not in hist.doc_judge_scores]
                if unjudged:
                    try:
                        extra_results, extra_usage = self.judge.judge_batch(
                            query, unjudged
                        )
                        self._accumulate_usage(hist, extra_usage)
                        for d, jr in zip(unjudged, extra_results):
                            rd.judge_results[d.doc_id] = jr
                            hist.seen_doc_ids.add(d.doc_id)
                            hist.doc_snippets[d.doc_id] = d.content
                            eff = jr.score if jr.score is not None else 0
                            if eff > hist.doc_judge_scores.get(d.doc_id, -1):
                                hist.doc_judge_scores[d.doc_id] = eff
                            if jr.parse_failed:
                                hist.doc_judge_parse_failed.add(d.doc_id)
                    except Exception as e:
                        logger.exception(
                            "Quality-saturation assessor error round %d qid %s: %s",
                            round_idx, query_id, e,
                        )

                all_scores = [
                    hist.doc_judge_scores.get(d.doc_id, 0)
                    for d in top_retrieved
                ]
                if all(s >= self.quality_saturation_min_score for s in all_scores):
                    hist.termination_reason = "quality_saturation"
                    break

            # ---- Coverage saturation check ---------------------------------
            reason, low_new_streak = self._termination_after_round(
                len(rd.new_doc_ids), low_new_streak
            )
            if reason is not None:
                hist.termination_reason = reason
                break
        else:
            hist.termination_reason = "round_budget"

        hist.usage["wall_time_seconds"] = round(time.time() - t0, 2)
        return hist

    # ----------------------------------------------------------- logging

    @staticmethod
    def _log_round(query_id: str, rd: Round) -> None:
        score_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        for jr in rd.judge_results.values():
            score_counts[jr.score if jr.score is not None else 0] += 1
        logger.info(
            "[qid=%s round=%d] retrieved=%d new_top10=%d new_pool=%d dupes=%d scores=%s "
            "exp_chars=%d refs=%d",
            query_id,
            rd.round_idx,
            len(rd.retrieved_doc_ids),
            len(rd.new_doc_ids),
            rd.n_new_docs_pool,
            rd.duplicate_count,
            score_counts,
            len(rd.expanded_query),
            len(rd.pseudo_refs),
        )


# ---------------------------------------------------------------------------


def _compute_ablation_fields(history: SearchHistory) -> dict[str, Any]:
    """Compute mode-agnostic ablation fields from the search history."""
    judge_dist = []
    retrieval_signal = []

    for rd in history.rounds:
        scores = rd.judge_results
        counts = {str(s): 0 for s in range(4)}
        score_vals: list[int] = []
        for jr in scores.values():
            s = jr.score if jr.score is not None else 0
            counts[str(s)] += 1
            score_vals.append(s)
        n_judged = len(scores)
        judge_dist.append({
            "round_idx": rd.round_idx,
            "n_judged": n_judged,
            "counts": counts,
            "mean": round(sum(score_vals) / max(n_judged, 1), 2),
        })

        ret_vals = sorted(rd.retrieval_scores.values(), reverse=True)
        top1 = ret_vals[0] if ret_vals else 0.0
        top10 = ret_vals[:10]
        top10_mean = sum(top10) / max(len(top10), 1)
        retrieval_signal.append({
            "round_idx": rd.round_idx,
            "top1_score": round(top1, 4),
            "top10_mean": round(top10_mean, 4),
            "score_gap_top1_top10": round(top1 - top10_mean, 4),
        })

    return {
        "total_rounds_executed": len(history.rounds),
        "judge_score_distribution_per_round": judge_dist,
        "retrieval_signal_per_round": retrieval_signal,
    }


def persist_trace(
    history: SearchHistory,
    path: str | Path,
    fusion_cfg: dict[str, Any] | None = None,
) -> None:
    """Dump a per-query trace JSON with ablation fields."""
    out = history.to_dict()
    out.update(_compute_ablation_fields(history))

    if fusion_cfg:
        from .fusion import fuse
        modes = fusion_cfg.get("modes_to_run", [])
        primary = fusion_cfg.get("primary_mode", "judge_plus_rrf")
        rankings_by_mode: dict[str, list[str]] = {}
        for mode in modes:
            ranked = fuse(history, mode, fusion_cfg)
            rankings_by_mode[mode] = [doc_id for doc_id, _, _ in ranked[:10]]
        out["final_rankings_top10_by_mode"] = rankings_by_mode
        out["final_ranking_top10"] = rankings_by_mode.get(
            primary, list(rankings_by_mode.values())[0] if rankings_by_mode else []
        )
    else:
        from .ranking import rank_history
        top = rank_history(history)[:10]
        out["final_ranking_top10"] = [doc_id for doc_id, _, _ in top]

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
