"""Final ranking construction.

Per query: union over all retrieved docs; sort by max assessor score descending,
break ties by best observed retrieval score descending. Output is a list of
`(doc_id, rank, score)` tuples ready for TREC-format dumping.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .fusion import fuse
from .schemas import SearchHistory


def _sanitize_doc_id(doc_id: str) -> str:
    """Replace whitespace with underscores so trec_eval can parse the 6-field TREC format."""
    return doc_id.replace(" ", "_")


def rank_history(history: SearchHistory) -> list[tuple[str, int, float]]:
    """Build the final ranking from the per-query trace.

    The TREC `score` column is `judge_score * 1000 + retrieval_score`, so docs
    ranked by assessor tier come first and retrieval score acts as a continuous
    tiebreaker. This keeps the scores strictly decreasing in rank, which
    `trec_eval` requires.
    """
    seen = history.seen_doc_ids
    if not seen:
        return []

    def composite(doc_id: str) -> float:
        j = history.doc_judge_scores.get(doc_id, 0)
        ret = history.doc_best_score.get(doc_id, 0.0)
        return float(j) * 1000.0 + ret

    sorted_ids = sorted(seen, key=composite, reverse=True)
    return [(doc_id, rank + 1, composite(doc_id)) for rank, doc_id in enumerate(sorted_ids)]


def write_trec_run(
    histories: list[SearchHistory], path: str, run_tag: str = "seek", top_k: int = 1000
) -> int:
    """Write all histories to a single TREC run file. Returns line count."""
    n_lines = 0
    with open(path, "w", encoding="utf-8") as f:
        for h in histories:
            ranked = rank_history(h)[:top_k]
            for doc_id, rank, score in ranked:
                f.write(f"{h.query_id} Q0 {_sanitize_doc_id(doc_id)} {rank} {score:.4f} {run_tag}\n")
                n_lines += 1
    return n_lines


def write_trec_run_fused(
    histories: list[SearchHistory],
    path: str | Path,
    mode: str,
    fusion_cfg: dict[str, Any],
    run_tag: str = "seek",
) -> int:
    """Write a TREC run file using a specific fusion mode. Returns line count."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n_lines = 0
    with open(path, "w", encoding="utf-8") as f:
        for h in histories:
            ranked = fuse(h, mode, fusion_cfg)
            for doc_id, rank, score in ranked:
                f.write(f"{h.query_id} Q0 {_sanitize_doc_id(doc_id)} {rank} {score:.4f} {run_tag}\n")
                n_lines += 1
    return n_lines
