"""Dataclasses shared by the SEEK agent, tools, and ranking modules."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    """A single retrieval hit.

    `content` is the (optionally truncated) text shown to the assessor;
    `raw` is the untruncated text, populated only when full traces are needed.
    """

    doc_id: str
    retrieval_score: float
    content: str
    raw: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d.pop("raw", None)  # keep traces compact
        return d


@dataclass
class JudgeResult:
    doc_id: str
    score: int | None              # 0/1/2/3, or None on parse failure (treated as 0)
    raw_output: str = ""
    model: str = ""
    parse_failed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class Round:
    round_idx: int                          # 1-indexed
    pseudo_refs: list[str] = field(default_factory=list)
    expanded_query: str = ""
    retrieved_doc_ids: list[str] = field(default_factory=list)
    new_doc_ids: list[str] = field(default_factory=list)  # new in top-10; assessed this round
    n_new_docs_pool: int = 0  # new in full retrieval list (for saturation / coverage)
    duplicate_count: int = 0
    judge_results: dict[str, JudgeResult] = field(default_factory=dict)
    retrieval_scores: dict[str, float] = field(default_factory=dict)
    generator_meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        top_ids = self.retrieved_doc_ids[:10]
        top10_id_set = set(top_ids)
        return {
            "round_idx": self.round_idx,
            "expanded_query_preview": self.expanded_query[:200],
            "expanded_query_chars": len(self.expanded_query),
            "pseudo_refs": self.pseudo_refs,
            "retrieved_doc_ids": top_ids,
            "new_doc_ids": self.new_doc_ids,
            "n_new_docs": self.n_new_docs_pool,
            "duplicate_count": self.duplicate_count,
            "judge_scores": {
                d: (j.score if j.score is not None else 0)
                for d, j in self.judge_results.items()
                if d in self.new_doc_ids
            },
            "judge_parse_failures": [
                d
                for d, j in self.judge_results.items()
                if j.parse_failed and d in self.new_doc_ids
            ],
            "retrieval_scores": {
                d: s for d, s in self.retrieval_scores.items() if d in top10_id_set
            },
            "generator_meta": self.generator_meta,
        }


@dataclass
class SearchHistory:
    """Per-query state accumulated across all SEEK retrieval rounds."""

    query_id: str
    original_query: str
    rounds: list[Round] = field(default_factory=list)
    seen_doc_ids: set[str] = field(default_factory=set)
    doc_judge_scores: dict[str, int] = field(default_factory=dict)
    doc_snippets: dict[str, str] = field(default_factory=dict)
    doc_best_score: dict[str, float] = field(default_factory=dict)
    doc_judge_parse_failed: set[str] = field(default_factory=set)
    termination_reason: str | None = None

    # Cost / usage accumulator
    usage: dict[str, float] = field(
        default_factory=lambda: {
            "llm_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "retrieval_calls": 0,
            "estimated_usd": 0.0,
        }
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "original_query": self.original_query,
            "rounds": [r.to_dict() for r in self.rounds],
            "termination_reason": self.termination_reason,
            "n_seen_docs": len(self.seen_doc_ids),
            "doc_judge_scores": self.doc_judge_scores,
            "doc_best_score": {
                d: s for d, s in self.doc_best_score.items()
                if d in self.doc_judge_scores
            },
            "doc_judge_parse_failed": sorted(self.doc_judge_parse_failed),
            "usage": self.usage,
        }


@dataclass
class PseudoPassages:
    """Pydantic-compatible shape for `chat.completions.parse` output."""

    passages: list[str]
