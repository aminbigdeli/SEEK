"""Alpha-repetition query expansion for pseudo-passage retrieval.

Combines the original query with generated pseudo-passages to form an
expanded query string. Two strategies are provided:

``build_expanded_query``
    The original query is repeated a number of times proportional to the
    ratio of pseudo-passage length to query length (controlled by ``alpha``),
    then followed by all pseudo-passages concatenated. This balances term
    frequency between the short original query and the longer generated text.

``build_passage_count_query``
    The original query is repeated once per pseudo-passage, then all
    pseudo-passages are appended. Used in combined / relevance-feedback modes.

QueryGym uses CHARACTER lengths (not tokens) and integer floor division:

    repetition_times = max(1, (docs_len // query_len) // alpha)
    expanded_query   = (query + ' ') * repetition_times + all_pseudo_passages

``alpha`` defaults to 5 following the standard configuration.
"""
from __future__ import annotations


def build_expanded_query(
    q0: str, pseudo_refs: list[str], alpha: int = 5
) -> tuple[str, dict]:
    """Build an expanded query by repeating `q0` to balance term frequencies
    against the much longer concatenated pseudo-passages.

    Returns
    -------
    (expanded_query, meta)
        ``meta`` carries the lengths and ``repetition_times`` for the trace.
    """
    q0 = q0.strip()
    all_pseudo_docs = " ".join(s.strip() for s in pseudo_refs if s and s.strip())

    query_len = len(q0)
    docs_len = len(all_pseudo_docs)

    if query_len > 0:
        repetition_times = max(1, (docs_len // query_len) // max(alpha, 1))
    else:
        repetition_times = 1

    expanded_query = (q0 + " ") * repetition_times + all_pseudo_docs

    meta = {
        "alpha": alpha,
        "query_chars": query_len,
        "docs_chars": docs_len,
        "repetition_times": repetition_times,
        "num_pseudo_refs": len(pseudo_refs),
    }
    return expanded_query, meta


def build_passage_count_query(
    q0: str, pseudo_refs: list[str]
) -> tuple[str, dict]:
    """Build an expanded query by repeating q0 once per pseudo-passage,
    then appending all passages."""
    q0 = q0.strip()
    passages = [s.strip() for s in pseudo_refs if s and s.strip()]
    all_pseudo_docs = " ".join(passages)
    repetition_times = len(passages)

    if repetition_times == 0:
        expanded_query = q0
    else:
        expanded_query = (q0 + " ") * repetition_times + all_pseudo_docs

    return expanded_query, {
        "alpha": None,
        "query_chars": len(q0),
        "docs_chars": len(all_pseudo_docs),
        "repetition_times": repetition_times,
        "num_pseudo_refs": len(passages),
    }
