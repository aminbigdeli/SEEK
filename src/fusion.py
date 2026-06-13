"""Multi-mode fusion: combine per-round retrieval rankings and assessor scores.

Six modes share the same RRF and bucket helpers, differing only in how they
compose the head (high-confidence) and tail (backfill) portions of the final
ranked list.  All expensive retrieval/assessing work is done once; this module
only reorders.
"""
from __future__ import annotations

from typing import Any

from .schemas import SearchHistory

MODES = (
    "judge_only_top_10",
    "judge_plus_rrf_top_10",
    "judge_plus_rrf",
    "judge_retrieval_head_rrf_tail",
    "judge_bucket_plus_rrf",
    "rrf_only",
    "quality_weighted_rrf",
)


# ---------------------------------------------------------------------------
# RRF helpers
# ---------------------------------------------------------------------------

def compute_rrf(
    history: SearchHistory,
    rrf_k: int = 60,
) -> dict[str, float]:
    """Standard Reciprocal Rank Fusion over per-round retrieved_doc_ids.

    Each round contributes 1/(k + rank) where rank is 1-indexed position in
    that round's retrieval result list.  Scores are normalized to [0, 1].
    """
    raw: dict[str, float] = {}
    for rd in history.rounds:
        for rank_0, doc_id in enumerate(rd.retrieved_doc_ids):
            raw[doc_id] = raw.get(doc_id, 0.0) + 1.0 / (rrf_k + rank_0 + 1)

    if not raw:
        return {}
    max_score = max(raw.values())
    if max_score == 0:
        return raw
    return {d: s / max_score for d, s in raw.items()}


def compute_quality_weighted_rrf(
    history: SearchHistory,
    rrf_k: int = 60,
) -> dict[str, float]:
    """RRF where each round's contribution is scaled by its mean assessor score.

    Rounds with higher-quality assessed documents get a bigger vote.
    """
    raw: dict[str, float] = {}
    for rd in history.rounds:
        scores = [
            (jr.score if jr.score is not None else 0)
            for jr in rd.judge_results.values()
        ]
        round_weight = (sum(scores) / max(len(scores), 1)) + 1e-6
        for rank_0, doc_id in enumerate(rd.retrieved_doc_ids):
            raw[doc_id] = raw.get(doc_id, 0.0) + round_weight / (rrf_k + rank_0 + 1)

    if not raw:
        return {}
    max_score = max(raw.values())
    if max_score == 0:
        return raw
    return {d: s / max_score for d, s in raw.items()}


# ---------------------------------------------------------------------------
# Bucket logic
# ---------------------------------------------------------------------------

_BUCKET_ORDER = {
    3: 60,
    2: 50,
    1: 40,
    "unjudged": 30,
    "0_demoted": 10,
    "0_dropped": -1,
}


def _bucket_tier(
    doc_id: str,
    judge_scores: dict[str, int],
    parse_failures: set[str],
    score_0_handling: str,
) -> int | str:
    """Return the bucket key for *doc_id*.

    - Score 1/2/3  → that integer
    - Parse failure → "unjudged"
    - Not assessed  → "unjudged"
    - Score 0       → "0_demoted" or "0_dropped" depending on config
    """
    if doc_id in parse_failures:
        return "unjudged"
    if doc_id not in judge_scores:
        return "unjudged"
    s = judge_scores[doc_id]
    if s == 0:
        return "0_demoted" if score_0_handling == "demote" else "0_dropped"
    return s


def _sort_key(tier: int | str, tiebreak: float) -> tuple[int, float]:
    order = _BUCKET_ORDER.get(tier, 20)
    return (order, tiebreak)


# ---------------------------------------------------------------------------
# Per-doc best retrieval score across rounds
# ---------------------------------------------------------------------------

def _best_retrieval_score(history: SearchHistory) -> dict[str, float]:
    best: dict[str, float] = {}
    for rd in history.rounds:
        for doc_id, score in rd.retrieval_scores.items():
            if score > best.get(doc_id, -1.0):
                best[doc_id] = score
    return best


# ---------------------------------------------------------------------------
# Pool of all doc_ids that appear anywhere across rounds
# ---------------------------------------------------------------------------

def _full_pool(history: SearchHistory) -> set[str]:
    pool: set[str] = set()
    for rd in history.rounds:
        pool.update(rd.retrieved_doc_ids)
    return pool


# ---------------------------------------------------------------------------
# Individual mode implementations
# ---------------------------------------------------------------------------

def _judge_only_top_10(
    history: SearchHistory, cfg: dict[str, Any],
) -> list[tuple[str, int, float]]:
    """Head: bucket + retrieval-score tiebreak, cap 10. Tail: retrieval score."""
    s0 = cfg.get("score_0_handling", "demote")
    head_cap = int(cfg.get("head_cap", 10))
    ret_scores = _best_retrieval_score(history)
    pool = _full_pool(history)

    scored = []
    for doc_id in pool:
        tier = _bucket_tier(doc_id, history.doc_judge_scores, history.doc_judge_parse_failed, s0)
        if tier == "0_dropped":
            continue
        scored.append((doc_id, tier, ret_scores.get(doc_id, 0.0)))

    scored.sort(key=lambda t: _sort_key(t[1], t[2]), reverse=True)

    head = scored[:head_cap]
    tail = scored[head_cap:]
    tail.sort(key=lambda t: t[2], reverse=True)
    return _assign_ranks(head + tail, cfg)


def _judge_plus_rrf_top_10(
    history: SearchHistory, cfg: dict[str, Any],
) -> list[tuple[str, int, float]]:
    """Head: bucket + RRF tiebreak, cap 10. Tail: retrieval score."""
    s0 = cfg.get("score_0_handling", "demote")
    head_cap = int(cfg.get("head_cap", 10))
    rrf_k = int(cfg.get("rrf_k", 60))
    rrf = compute_rrf(history, rrf_k)
    ret_scores = _best_retrieval_score(history)
    pool = _full_pool(history)

    judged_positive = []
    rest = []
    for doc_id in pool:
        tier = _bucket_tier(doc_id, history.doc_judge_scores, history.doc_judge_parse_failed, s0)
        if tier == "0_dropped":
            continue
        entry = (doc_id, tier, rrf.get(doc_id, 0.0), ret_scores.get(doc_id, 0.0))
        if isinstance(tier, int) and tier > 0:
            judged_positive.append(entry)
        else:
            rest.append(entry)

    judged_positive.sort(key=lambda t: _sort_key(t[1], t[2]), reverse=True)
    head_candidates = judged_positive[:head_cap]

    if len(head_candidates) < head_cap:
        unjudged = [e for e in rest if e[1] == "unjudged"]
        unjudged.sort(key=lambda t: t[2], reverse=True)
        head_candidates.extend(unjudged[: head_cap - len(head_candidates)])

    head_ids = {e[0] for e in head_candidates}
    tail = [e for e in judged_positive + rest if e[0] not in head_ids]
    tail.sort(key=lambda t: t[3], reverse=True)

    combined = [(d, t, r) for d, t, r, _b in head_candidates] + [
        (d, t, b) for d, t, _r, b in tail
    ]
    return _assign_ranks(combined, cfg)


def _judge_plus_rrf(
    history: SearchHistory, cfg: dict[str, Any],
) -> list[tuple[str, int, float]]:
    """Head: bucket + RRF tiebreak, cap 10. Tail: RRF."""
    s0 = cfg.get("score_0_handling", "demote")
    head_cap = int(cfg.get("head_cap", 10))
    rrf_k = int(cfg.get("rrf_k", 60))
    rrf = compute_rrf(history, rrf_k)
    pool = _full_pool(history)

    judged_positive = []
    rest = []
    for doc_id in pool:
        tier = _bucket_tier(doc_id, history.doc_judge_scores, history.doc_judge_parse_failed, s0)
        if tier == "0_dropped":
            continue
        entry = (doc_id, tier, rrf.get(doc_id, 0.0))
        if isinstance(tier, int) and tier > 0:
            judged_positive.append(entry)
        else:
            rest.append(entry)

    judged_positive.sort(key=lambda t: _sort_key(t[1], t[2]), reverse=True)
    head_candidates = list(judged_positive[:head_cap])

    if len(head_candidates) < head_cap:
        unjudged = [e for e in rest if e[1] == "unjudged"]
        unjudged.sort(key=lambda t: t[2], reverse=True)
        head_candidates.extend(unjudged[: head_cap - len(head_candidates)])

    head_ids = {e[0] for e in head_candidates}
    tail = [e for e in judged_positive + rest if e[0] not in head_ids]
    tail.sort(key=lambda t: t[2], reverse=True)

    return _assign_ranks(head_candidates + tail, cfg)


def _judge_retrieval_head_rrf_tail(
    history: SearchHistory, cfg: dict[str, Any],
) -> list[tuple[str, int, float]]:
    """Head: bucket + retrieval-score tiebreak, cap 10. Tail: RRF."""
    s0 = cfg.get("score_0_handling", "demote")
    head_cap = int(cfg.get("head_cap", 10))
    rrf_k = int(cfg.get("rrf_k", 60))
    rrf = compute_rrf(history, rrf_k)
    ret_scores = _best_retrieval_score(history)
    pool = _full_pool(history)

    judged_positive = []
    rest = []
    for doc_id in pool:
        tier = _bucket_tier(doc_id, history.doc_judge_scores, history.doc_judge_parse_failed, s0)
        if tier == "0_dropped":
            continue
        entry = (doc_id, tier, ret_scores.get(doc_id, 0.0), rrf.get(doc_id, 0.0))
        if isinstance(tier, int) and tier > 0:
            judged_positive.append(entry)
        else:
            rest.append(entry)

    judged_positive.sort(key=lambda t: _sort_key(t[1], t[2]), reverse=True)
    head_candidates = judged_positive[:head_cap]

    if len(head_candidates) < head_cap:
        unjudged = [e for e in rest if e[1] == "unjudged"]
        unjudged.sort(key=lambda t: t[2], reverse=True)
        head_candidates.extend(unjudged[: head_cap - len(head_candidates)])

    head_ids = {e[0] for e in head_candidates}
    tail = [e for e in judged_positive + rest if e[0] not in head_ids]
    tail.sort(key=lambda t: t[3], reverse=True)

    combined = [(d, t, ret) for d, t, ret, _r in head_candidates] + [
        (d, t, r) for d, t, _ret, r in tail
    ]
    return _assign_ranks(combined, cfg)


def _judge_bucket_plus_rrf(
    history: SearchHistory, cfg: dict[str, Any],
) -> list[tuple[str, int, float]]:
    """Head: ALL assessed-positive by bucket + RRF (no cap). Tail: RRF."""
    s0 = cfg.get("score_0_handling", "demote")
    rrf_k = int(cfg.get("rrf_k", 60))
    rrf = compute_rrf(history, rrf_k)
    pool = _full_pool(history)

    judged_positive = []
    rest = []
    for doc_id in pool:
        tier = _bucket_tier(doc_id, history.doc_judge_scores, history.doc_judge_parse_failed, s0)
        if tier == "0_dropped":
            continue
        entry = (doc_id, tier, rrf.get(doc_id, 0.0))
        if isinstance(tier, int) and tier > 0:
            judged_positive.append(entry)
        else:
            rest.append(entry)

    judged_positive.sort(key=lambda t: _sort_key(t[1], t[2]), reverse=True)
    rest.sort(key=lambda t: t[2], reverse=True)

    return _assign_ranks(judged_positive + rest, cfg)


def _rrf_only(
    history: SearchHistory, cfg: dict[str, Any],
) -> list[tuple[str, int, float]]:
    """Entire pool ranked by RRF only (assessor scores ignored)."""
    rrf_k = int(cfg.get("rrf_k", 60))
    rrf = compute_rrf(history, rrf_k)
    pool = _full_pool(history)
    entries = [(doc_id, 0, rrf.get(doc_id, 0.0)) for doc_id in pool]
    entries.sort(key=lambda t: t[2], reverse=True)
    return _assign_ranks(entries, cfg)


def _quality_weighted_rrf(
    history: SearchHistory, cfg: dict[str, Any],
) -> list[tuple[str, int, float]]:
    """Entire pool ranked by quality-weighted RRF."""
    rrf_k = int(cfg.get("rrf_k", 60))
    qrrf = compute_quality_weighted_rrf(history, rrf_k)
    pool = _full_pool(history)
    entries = [(doc_id, 0, qrrf.get(doc_id, 0.0)) for doc_id in pool]
    entries.sort(key=lambda t: t[2], reverse=True)
    return _assign_ranks(entries, cfg)


# ---------------------------------------------------------------------------
# Assign 1-indexed ranks and cap output
# ---------------------------------------------------------------------------

def _assign_ranks(
    items: list[tuple[str, int | str, float]],
    cfg: dict[str, Any],
) -> list[tuple[str, int, float]]:
    top_k = int(cfg.get("top_k_output", 1000))
    result: list[tuple[str, int, float]] = []
    for rank_0, (doc_id, _tier, score) in enumerate(items[:top_k]):
        result.append((doc_id, rank_0 + 1, score))
    return result


# ---------------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------------

_MODE_DISPATCH = {
    "judge_only_top_10": _judge_only_top_10,
    "judge_plus_rrf_top_10": _judge_plus_rrf_top_10,
    "judge_plus_rrf": _judge_plus_rrf,
    "judge_retrieval_head_rrf_tail": _judge_retrieval_head_rrf_tail,
    "judge_bucket_plus_rrf": _judge_bucket_plus_rrf,
    "rrf_only": _rrf_only,
    "quality_weighted_rrf": _quality_weighted_rrf,
}


def fuse(
    history: SearchHistory,
    mode: str,
    cfg: dict[str, Any],
) -> list[tuple[str, int, float]]:
    """Produce a ranked list for *mode* from a completed SearchHistory.

    Returns list of (doc_id, 1-indexed rank, score).
    """
    fn = _MODE_DISPATCH.get(mode)
    if fn is None:
        raise ValueError(f"Unknown fusion mode: {mode!r}. Choose from {MODES}")
    return fn(history, cfg)
