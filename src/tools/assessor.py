"""UMBRELA-based LLM relevance assessor.

One LLM call per (query, passage) pair, using the UMBRELA prompt template.
Score parsed from the `##final score: N` line. Parallel fan-out within a
round via `concurrent.futures.ThreadPoolExecutor`.
"""
from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml

from ..llm_client import LLMClient, LLMResult
from ..schemas import Document, JudgeResult

logger = logging.getLogger(__name__)


_SCORE_RE = re.compile(r"##\s*final\s+score\s*:\s*([0-3])", re.IGNORECASE)


def _load_umbrela_template(path: str | Path) -> str:
    """Load `prefix_user` from the UMBRELA YAML template."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    tmpl = data.get("prefix_user") or data.get("user") or ""
    if not tmpl:
        raise RuntimeError(f"UMBRELA template missing prefix_user in {path}")
    return tmpl


def parse_score(text: str) -> int | None:
    """Extract the integer score from an UMBRELA response. Returns None on miss."""
    m = _SCORE_RE.search(text or "")
    if not m:
        return None
    try:
        s = int(m.group(1))
    except ValueError:
        return None
    return s if 0 <= s <= 3 else None


class Judge:
    """Score retrieved docs against the ORIGINAL query using UMBRELA."""

    def __init__(
        self,
        llm: LLMClient,
        model: str,
        umbrela_template_path: str | Path,
        temperature: float = 0.0,
        max_tokens: int = 256,
        num_threads: int = 4,
    ):
        self.llm = llm
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.num_threads = max(1, num_threads)
        self.template = _load_umbrela_template(umbrela_template_path)
        self.system_message = ""  # UMBRELA template ships an empty system

    # ---------------------------------------------------------- internal

    def _render_messages(self, query: str, passage: str) -> list[dict[str, str]]:
        # Static prefix (instructions) first, variable bits last — helps
        # OpenAI's automatic prompt prefix caching.
        prompt = self.template.format(query=query, passage=passage)
        return [{"role": "user", "content": prompt}]

    def _judge_one(self, query: str, doc: Document) -> JudgeResult:
        messages = self._render_messages(query, doc.content)
        res = self.llm.chat_text(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            cache_namespace="judge_umbrela",
        )
        score = parse_score(res.text)
        if score is None:
            # One retry on parse failure
            res2 = self.llm.chat_text(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                cache_namespace="judge_umbrela_retry",
            )
            score = parse_score(res2.text)
            return JudgeResult(
                doc_id=doc.doc_id,
                score=score,
                raw_output=res.text + "\n----RETRY----\n" + res2.text,
                model=self.model,
                parse_failed=(score is None),
            )
        return JudgeResult(
            doc_id=doc.doc_id,
            score=score,
            raw_output=res.text,
            model=self.model,
            parse_failed=False,
        )

    # ------------------------------------------------------------ public

    def judge_batch(
        self, original_query: str, docs: list[Document]
    ) -> tuple[list[JudgeResult], dict[str, Any]]:
        """Assess all `docs` in parallel against `original_query`.

        Returns the results in the same order as `docs`, plus a usage dict
        suitable for accumulation into `SearchHistory.usage`.
        """
        if not docs:
            return [], {"llm_calls": 0, "input_tokens": 0, "output_tokens": 0, "estimated_usd": 0.0}

        results: dict[str, JudgeResult] = {}
        with ThreadPoolExecutor(max_workers=self.num_threads) as ex:
            future_to_doc = {
                ex.submit(self._judge_one, original_query, d): d for d in docs
            }
            for fut in as_completed(future_to_doc):
                d = future_to_doc[fut]
                try:
                    results[d.doc_id] = fut.result()
                except Exception as e:
                    logger.warning("Assessor failed for %s: %s", d.doc_id, e)
                    results[d.doc_id] = JudgeResult(
                        doc_id=d.doc_id,
                        score=None,
                        raw_output=f"ERROR: {e}",
                        model=self.model,
                        parse_failed=True,
                    )

        ordered = [results[d.doc_id] for d in docs]

        usage = {"llm_calls": len(ordered), "input_tokens": 0, "output_tokens": 0, "estimated_usd": 0.0}
        for d in docs:
            messages = self._render_messages(original_query, d.content)
            res = self.llm.chat_text(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                cache_namespace="judge_umbrela",
            )
            usage["input_tokens"] += res.input_tokens
            usage["output_tokens"] += res.output_tokens
            usage["estimated_usd"] += res.usd

        return ordered, usage
