"""Pyserini BM25 retriever.

Encapsulates index lookup, returns a list of ``Document`` objects with scores
stored in ``retrieval_score``, and caches results on disk so reruns are free.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from ..schemas import Document

logger = logging.getLogger(__name__)


def _load_pyserini():
    try:
        from pyserini.search.lucene import LuceneSearcher  # noqa: F401
        return LuceneSearcher
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Pyserini is required for BM25 retrieval. "
            "Install with `pip install pyserini>=0.22.0` and ensure a JDK is on PATH."
        ) from e


def _truncate_tokens(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return text
    parts = text.split()
    if len(parts) <= max_tokens:
        return text
    return " ".join(parts[:max_tokens])


def _parse_doc_raw(raw: str) -> str:
    if not raw:
        return ""
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            for key in ("contents", "text", "passage", "body"):
                v = obj.get(key)
                if isinstance(v, str) and v.strip():
                    return v
        return raw
    except Exception:
        return raw


class BM25Retriever:
    """BM25 retriever over a Pyserini Lucene index."""

    def __init__(
        self,
        index_path: str,
        prebuilt: bool = False,
        bm25_k1: float = 0.9,
        bm25_b: float = 0.4,
        top_k: int = 1000,
        snippet_max_tokens: int = 500,
        cache_dir: str | Path | None = ".cache/bm25",
    ):
        LuceneSearcher = _load_pyserini()
        if prebuilt:
            logger.info("Loading prebuilt Pyserini index: %s", index_path)
            self.searcher = LuceneSearcher.from_prebuilt_index(index_path)
        else:
            logger.info("Loading local Pyserini index: %s", index_path)
            self.searcher = LuceneSearcher(index_path)
        self.searcher.set_bm25(k1=bm25_k1, b=bm25_b)
        self.top_k = top_k
        self.snippet_max_tokens = snippet_max_tokens
        self.index_id = index_path

        self._cache_dir: Path | None
        if cache_dir is None:
            self._cache_dir = None
        else:
            self._cache_dir = Path(cache_dir)
            self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_key(self, query: str, top_k: int) -> str:
        return hashlib.sha256(
            f"{self.index_id}|{top_k}|{query}".encode("utf-8")
        ).hexdigest()

    def _cache_path(self, key: str) -> Path | None:
        if self._cache_dir is None:
            return None
        return self._cache_dir / f"{key}.json"

    def _load_cache(self, key: str) -> list[Document] | None:
        path = self._cache_path(key)
        if path is None or not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            return [
                Document(
                    doc_id=d["doc_id"],
                    retrieval_score=d["retrieval_score"],
                    content=d["content"],
                )
                for d in payload
            ]
        except Exception:
            return None

    def _save_cache(self, key: str, docs: list[Document]) -> None:
        path = self._cache_path(key)
        if path is None:
            return
        try:
            with path.open("w", encoding="utf-8") as f:
                json.dump(
                    [
                        {
                            "doc_id": d.doc_id,
                            "retrieval_score": d.retrieval_score,
                            "content": d.content,
                        }
                        for d in docs
                    ],
                    f,
                )
        except Exception:
            pass

    def _fetch_doc_text(self, doc_id: str) -> str:
        try:
            raw_doc = self.searcher.doc(doc_id)
            if raw_doc is None:
                return ""
            raw = raw_doc.raw() or ""
            return _parse_doc_raw(raw)
        except Exception:
            return ""

    def search(self, query: str, top_k: int | None = None) -> list[Document]:
        k = top_k or self.top_k
        key = self._cache_key(query, k)
        cached = self._load_cache(key)
        if cached is not None:
            return cached

        hits = self.searcher.search(query, k=k)
        docs: list[Document] = []
        for hit in hits:
            raw_text = self._fetch_doc_text(hit.docid)
            snippet = _truncate_tokens(raw_text, self.snippet_max_tokens)
            docs.append(
                Document(
                    doc_id=hit.docid,
                    retrieval_score=float(hit.score),
                    content=snippet,
                    raw=raw_text,
                )
            )
        self._save_cache(key, docs)
        return docs
