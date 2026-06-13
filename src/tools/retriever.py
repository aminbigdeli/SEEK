"""Dense retriever backed by pre-computed corpus embeddings (ReasonIR-8B).

Public interface:

    retriever = DenseRetriever(...)
    docs: list[Document] = retriever.search(query)

Corpus embeddings are pre-computed offline by ``scripts/embed_corpus.py``
and stored as:

    {embeddings_dir}/{benchmark}/
        embeddings.npy      float32 (N, dim) — one row per document
        doc_ids.json        ordered list of N doc-id strings
        doc_texts.jsonl     {"id": ..., "text": ...} lines for assessor snippets

At search time the query is encoded with the task-specific instruction,
cosine similarity is computed against all corpus embeddings, and the top-k
Documents are returned with the cosine similarity stored in the
``retrieval_score`` field (transparent to ``runner.py`` and ``ranking.py``).
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..schemas import Document

logger = logging.getLogger(__name__)


def _truncate_tokens(text: str, max_tokens: int) -> str:
    """Crude whitespace-token truncation."""
    if max_tokens <= 0:
        return text
    parts = text.split()
    if len(parts) <= max_tokens:
        return text
    return " ".join(parts[:max_tokens])


class DenseRetriever:
    """Dense retriever backed by pre-computed ReasonIR-8B corpus embeddings.

    Parameters
    ----------
    embeddings_path:
        Path to ``embeddings.npy`` — float32 array of shape (N, dim).
    doc_ids_path:
        Path to ``doc_ids.json`` — JSON list of N doc-id strings in the same
        row order as the embeddings.
    doc_texts_path:
        Path to ``doc_texts.jsonl`` — one JSON object per line with ``id``
        and ``text`` keys, used to populate ``Document.content``.
    model_path:
        HuggingFace model id or local directory for ReasonIR-8B.
    query_instruction:
        Task-specific instruction prepended to every query.
    top_k:
        Default number of documents to return per search call.
    snippet_max_tokens:
        Crude whitespace-token limit applied to ``Document.content``.
    cache_dir:
        If set, query embeddings are cached on disk (SHA-256 keyed) so
        repeated queries in the same run are free.
    encode_batch_size:
        Batch size passed to ``model.encode()``.
    max_query_length:
        Maximum token length for query encoding.
    device:
        ``"cuda"``, ``"cpu"``, or ``None`` for auto-detect.
    """

    def __init__(
        self,
        embeddings_path: str | Path,
        doc_ids_path: str | Path,
        doc_texts_path: str | Path,
        model_path: str = "reasonir/ReasonIR-8B",
        query_instruction: str = (
            "<|user|>\nGiven a query, retrieve relevant passages "
            "that help answer the query\n<|embed|>\n"
        ),
        top_k: int = 10,
        snippet_max_tokens: int = 500,
        cache_dir: str | Path | None = ".cache/seek",
        encode_batch_size: int = 1,
        max_query_length: int = 512,
        device: str | None = None,
    ):
        self.top_k = top_k
        self.snippet_max_tokens = snippet_max_tokens
        self.query_instruction = query_instruction
        self.encode_batch_size = encode_batch_size
        self.max_query_length = max_query_length

        if device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = device
        logger.info("DenseRetriever using device=%s", self._device)

        logger.info("Loading model from %s ...", model_path)
        from transformers import AutoModel
        self._model = AutoModel.from_pretrained(
            model_path,
            torch_dtype="auto",
            trust_remote_code=True,
        )
        self._model.eval()
        self._model.to(self._device)
        logger.info("Model loaded.")

        embeddings_path = Path(embeddings_path)
        doc_ids_path = Path(doc_ids_path)
        doc_texts_path = Path(doc_texts_path)

        logger.info("Loading corpus embeddings from %s ...", embeddings_path)
        emb_np = np.load(str(embeddings_path)).astype(np.float32)
        emb_t = torch.from_numpy(emb_np)
        self._doc_emb: torch.Tensor = F.normalize(emb_t, p=2, dim=1).to(self._device)
        logger.info(
            "Corpus: %d documents, dim=%d",
            self._doc_emb.shape[0],
            self._doc_emb.shape[1],
        )

        with doc_ids_path.open("r", encoding="utf-8") as f:
            self._doc_ids: list[str] = json.load(f)

        self._doc_texts: dict[str, str] = {}
        with doc_texts_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                self._doc_texts[obj["id"]] = obj.get("text", "")

        assert len(self._doc_ids) == self._doc_emb.shape[0], (
            f"doc_ids ({len(self._doc_ids)}) and embeddings "
            f"({self._doc_emb.shape[0]}) row counts do not match."
        )

        self._cache_dir: Path | None
        if cache_dir is not None:
            self._cache_dir = Path(cache_dir)
            self._cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self._cache_dir = None

        logger.info("DenseRetriever ready (top_k=%d).", top_k)

    # ── Cache helpers ──────────────────────────────────────────────────────

    def _cache_key(self, query: str, k: int) -> str:
        payload = f"{self.query_instruction}|{k}|{query}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> Path | None:
        if self._cache_dir is None:
            return None
        return self._cache_dir / f"{key}.json"

    def _load_cache(self, key: str) -> list[Document] | None:
        p = self._cache_path(key)
        if p is None or not p.exists():
            return None
        try:
            with p.open("r", encoding="utf-8") as f:
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
        p = self._cache_path(key)
        if p is None:
            return
        try:
            with p.open("w", encoding="utf-8") as f:
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

    # ── Public interface ───────────────────────────────────────────────────

    def search(self, query: str, top_k: int | None = None) -> list[Document]:
        """Encode *query* and return the top-k most similar documents.

        Cosine similarity is stored in ``Document.retrieval_score``.
        """
        k = top_k if top_k is not None else self.top_k
        key = self._cache_key(query, k)
        cached = self._load_cache(key)
        if cached is not None:
            return cached

        with torch.inference_mode():
            q_emb = self._model.encode(
                [query],
                instruction=self.query_instruction,
                batch_size=self.encode_batch_size,
                max_length=self.max_query_length,
            )
            q_t = F.normalize(
                torch.from_numpy(q_emb.astype(np.float32)), p=2, dim=1
            ).to(self._device)

        scores = (q_t @ self._doc_emb.T).squeeze(0)

        actual_k = min(k, scores.shape[0])
        top_scores, top_indices = torch.topk(scores, actual_k)

        docs: list[Document] = []
        for idx, score in zip(top_indices.tolist(), top_scores.tolist()):
            doc_id = self._doc_ids[idx]
            raw_text = self._doc_texts.get(doc_id, "")
            content = _truncate_tokens(raw_text, self.snippet_max_tokens)
            docs.append(
                Document(doc_id=doc_id, retrieval_score=float(score), content=content)
            )

        self._save_cache(key, docs)
        return docs
