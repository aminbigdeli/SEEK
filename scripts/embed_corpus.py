"""Pre-compute corpus embeddings for SEEK using ReasonIR-8B.

Encodes all documents in a benchmark corpus and saves:

    {output_dir}/
        embeddings.npy      float32 (N, dim) — one row per document
        doc_ids.json        ordered list of N doc-id strings
        doc_texts.jsonl     {"id": ..., "text": ...} for assessor snippet lookup

Corpus sources (choose one):
  --benchmark       Load from HuggingFace ``xlangai/BRIGHT`` (for BRIGHT tasks)
  --corpus_jsonl    Load from a BEIR-style JSONL ({"_id": ..., "text": ...})
  --pyserini_index  Load from a Pyserini Lucene index (extracts stored text)

Usage examples
--------------
# BRIGHT biology (downloads ~50 MB from HuggingFace)
python scripts/embed_corpus.py \\
    --benchmark bright-biology \\
    --output_dir .cache/seek_embeddings/bright-biology \\
    --model_path reasonir/ReasonIR-8B \\
    --batch_size 8 --max_doc_length 2048

# Custom BEIR corpus
python scripts/embed_corpus.py \\
    --corpus_jsonl /path/to/corpus.jsonl \\
    --output_dir .cache/seek_embeddings/my-corpus \\
    --model_path reasonir/ReasonIR-8B
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("embed_corpus")

_HF_DATASET = "xlangai/BRIGHT"

_BRIGHT_TASK_MAP: dict[str, str] = {
    "bright-biology": "biology",
    "bright-earth-science": "earth_science",
    "bright-economics": "economics",
    "bright-psychology": "psychology",
    "bright-robotics": "robotics",
    "bright-stackoverflow": "stackoverflow",
    "bright-sustainable-living": "sustainable_living",
    "bright-pony": "pony",
    "bright-leetcode": "leetcode",
    "bright-aops": "aops",
    "bright-theoremqa-theorems": "theoremqa_theorems",
    "bright-theoremqa-questions": "theoremqa_questions",
}

DOC_INSTRUCTION = "<|embed|>\n"


# ---------------------------------------------------------------------------
# Corpus loaders
# ---------------------------------------------------------------------------

def load_corpus_bright(benchmark_key: str, cache_dir: str | None = None) -> list[dict]:
    """Load documents from HuggingFace xlangai/BRIGHT."""
    task = _BRIGHT_TASK_MAP.get(benchmark_key)
    if task is None:
        raise SystemExit(
            f"Unknown BRIGHT benchmark key '{benchmark_key}'. "
            f"Known keys: {sorted(_BRIGHT_TASK_MAP)}"
        )
    logger.info("Loading BRIGHT corpus '%s' from HuggingFace ...", task)
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("HuggingFace datasets not installed. pip install datasets")
    hf_kwargs: dict = {}
    if cache_dir:
        hf_kwargs["cache_dir"] = cache_dir
    ds = load_dataset(_HF_DATASET, "documents", **hf_kwargs)
    docs_raw = ds[task]
    docs = []
    for ex in docs_raw:
        doc_id = str(ex.get("id") or ex.get("_id") or "")
        text_field = ex.get("content") or ex.get("text") or ex.get("contents") or ex.get(task) or ""
        if doc_id and text_field:
            docs.append({"id": doc_id, "text": str(text_field)})
    logger.info("Loaded %d documents.", len(docs))
    return docs


def load_corpus_jsonl(path: str) -> list[dict]:
    """Load documents from a BEIR-style JSONL file."""
    logger.info("Loading corpus from JSONL: %s ...", path)
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            doc_id = str(
                obj.get("_id") or obj.get("id") or obj.get("doc_id") or ""
            )
            text = (
                obj.get("text")
                or obj.get("contents")
                or obj.get("passage")
                or obj.get("body")
                or ""
            )
            if isinstance(text, dict):
                text = (obj.get("title") or "") + " " + str(text.get("text", ""))
            title = obj.get("title", "")
            if title and not str(text).startswith(str(title)):
                text = f"{title} {text}"
            if doc_id and text:
                docs.append({"id": doc_id, "text": str(text).strip()})
    logger.info("Loaded %d documents.", len(docs))
    return docs


def load_corpus_pyserini(index_path: str, prebuilt: bool = False) -> list[dict]:
    """Extract documents from a Pyserini Lucene index."""
    logger.info("Loading corpus from Pyserini index: %s ...", index_path)
    try:
        from pyserini.search.lucene import LuceneSearcher
    except ImportError:
        raise SystemExit("Pyserini not installed. pip install pyserini>=0.22.0")
    if prebuilt:
        s = LuceneSearcher.from_prebuilt_index(index_path)
    else:
        s = LuceneSearcher(index_path)

    num_docs = s.num_docs
    logger.info("Index has %d documents. Extracting ...", num_docs)
    docs = []
    for i in range(num_docs):
        try:
            raw_doc = s.doc(i)
            if raw_doc is None:
                continue
            doc_id = raw_doc.docid()
            raw = raw_doc.raw() or ""
            try:
                obj = json.loads(raw)
                if isinstance(obj, dict):
                    text = (
                        obj.get("contents")
                        or obj.get("text")
                        or obj.get("passage")
                        or obj.get("body")
                        or raw
                    )
                else:
                    text = raw
            except Exception:
                text = raw
            if doc_id and text:
                docs.append({"id": str(doc_id), "text": str(text)})
        except Exception:
            continue
        if (i + 1) % 10000 == 0:
            logger.info("  Extracted %d / %d docs ...", i + 1, num_docs)
    logger.info("Loaded %d documents.", len(docs))
    return docs


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embed_documents(
    docs: list[dict],
    model_path: str,
    batch_size: int,
    max_doc_length: int,
    output_dir: Path,
    device: str | None = None,
) -> None:
    """Encode documents and save embeddings.npy, doc_ids.json, doc_texts.jsonl."""
    output_dir.mkdir(parents=True, exist_ok=True)

    emb_path = output_dir / "embeddings.npy"
    ids_path = output_dir / "doc_ids.json"
    texts_path = output_dir / "doc_texts.jsonl"

    if emb_path.exists() and ids_path.exists() and texts_path.exists():
        logger.info("Embeddings already exist at %s — skipping.", output_dir)
        logger.info("Pass --force to overwrite or delete the directory manually.")
        return

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading model %s on %s ...", model_path, device)

    from transformers import AutoModel
    model = AutoModel.from_pretrained(
        model_path, torch_dtype="auto", trust_remote_code=True
    )
    model.eval()
    model.to(device)
    logger.info("Model loaded.")

    texts = [d["text"] for d in docs]
    doc_ids = [d["id"] for d in docs]
    n = len(texts)
    logger.info("Encoding %d documents in batches of %d ...", n, batch_size)

    all_embs: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, n, batch_size):
            batch = texts[start : start + batch_size]
            emb = model.encode(
                batch,
                instruction=DOC_INSTRUCTION,
                batch_size=batch_size,
                max_length=max_doc_length,
            )
            all_embs.append(emb.astype(np.float32))
            if (start // batch_size + 1) % 20 == 0 or start + batch_size >= n:
                logger.info(
                    "  Encoded %d / %d documents ...", min(start + batch_size, n), n
                )

    embeddings = np.concatenate(all_embs, axis=0)
    logger.info("Embeddings shape: %s", embeddings.shape)

    np.save(str(emb_path), embeddings)
    logger.info("Saved %s", emb_path)

    with ids_path.open("w", encoding="utf-8") as f:
        json.dump(doc_ids, f)
    logger.info("Saved %s", ids_path)

    with texts_path.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps({"id": d["id"], "text": d["text"]}, ensure_ascii=False) + "\n")
    logger.info("Saved %s", texts_path)
    logger.info("Done. %d documents embedded.", n)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pre-compute corpus embeddings for SEEK."
    )
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--benchmark",
        metavar="KEY",
        help="BRIGHT registry key (e.g. bright-biology). Downloads from HuggingFace.",
    )
    source.add_argument(
        "--corpus_jsonl",
        metavar="PATH",
        help="Path to a BEIR-style JSONL corpus file.",
    )
    source.add_argument(
        "--pyserini_index",
        metavar="INDEX",
        help="Pyserini index name (prebuilt) or path (local).",
    )
    p.add_argument(
        "--output_dir",
        required=True,
        metavar="DIR",
        help="Directory to write embeddings.npy, doc_ids.json, doc_texts.jsonl.",
    )
    p.add_argument(
        "--model_path",
        default="reasonir/ReasonIR-8B",
        help="HuggingFace model ID or local path (default: reasonir/ReasonIR-8B).",
    )
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument(
        "--max_doc_length", type=int, default=2048,
        help="Maximum token length for document encoding."
    )
    p.add_argument(
        "--prebuilt",
        action="store_true",
        help="When using --pyserini_index, treat it as a Pyserini prebuilt index name.",
    )
    p.add_argument(
        "--hf_cache_dir",
        default=None,
        help="HuggingFace datasets cache directory (for --benchmark).",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing embeddings.",
    )
    p.add_argument("--device", default=None, help="cuda / cpu (default: auto).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    if args.force:
        for fname in ("embeddings.npy", "doc_ids.json", "doc_texts.jsonl"):
            p = output_dir / fname
            if p.exists():
                p.unlink()
                logger.info("Removed %s", p)

    if args.benchmark:
        docs = load_corpus_bright(args.benchmark, cache_dir=args.hf_cache_dir)
    elif args.corpus_jsonl:
        docs = load_corpus_jsonl(args.corpus_jsonl)
    else:
        docs = load_corpus_pyserini(args.pyserini_index, prebuilt=args.prebuilt)

    if not docs:
        logger.error("No documents loaded — check your corpus source.")
        sys.exit(1)

    embed_documents(
        docs=docs,
        model_path=args.model_path,
        batch_size=args.batch_size,
        max_doc_length=args.max_doc_length,
        output_dir=output_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
