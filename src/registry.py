"""Dataset registry: resolves a benchmark key into the concrete
(Pyserini index, topics, qrels, BM25 weights, default metrics).

Mirrors QueryGym's `dataset_registry.yaml` schema so single-shot retrieval is
directly comparable to their leaderboard. Pyserini topic/qrels names (e.g.
`dl19-passage`, `beir-v1.0.0-scifact-test`) are loaded via
`pyserini.search.get_topics` / `get_qrels`, so we never hand-roll the
mappings.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


# Short user-facing aliases mapped to canonical registry keys.
_ALIASES = {
    # TREC DL passage
    "trec-dl-2019": "msmarco-v1-passage.trecdl2019",
    "trec-dl-2020": "msmarco-v1-passage.trecdl2020",
    "trec-dl19": "msmarco-v1-passage.trecdl2019",
    "trec-dl20": "msmarco-v1-passage.trecdl2020",
    "dl19": "msmarco-v1-passage.trecdl2019",
    "dl20": "msmarco-v1-passage.trecdl2020",
    "dl-hard": "dl-hard",
    "dlhard": "dl-hard",
    "msmarco-dev": "msmarco-v1-passage.dev",
    # BEIR conveniences (drop the "beir-v1.0.0-" prefix)
    "scifact": "beir-v1.0.0-scifact",
    "trec-covid": "beir-v1.0.0-trec-covid",
    "nfcorpus": "beir-v1.0.0-nfcorpus",
    "nq": "beir-v1.0.0-nq",
    "hotpotqa": "beir-v1.0.0-hotpotqa",
    "fiqa": "beir-v1.0.0-fiqa",
    "arguana": "beir-v1.0.0-arguana",
    "trec-news": "beir-v1.0.0-trec-news",
    "robust04": "beir-v1.0.0-robust04",
    "webis-touche2020": "beir-v1.0.0-webis-touche2020",
    "quora": "beir-v1.0.0-quora",
    "dbpedia-entity": "beir-v1.0.0-dbpedia-entity",
    "scidocs": "beir-v1.0.0-scidocs",
    "fever": "beir-v1.0.0-fever",
    "climate-fever": "beir-v1.0.0-climate-fever",
    "bioasq": "beir-v1.0.0-bioasq",
    "signal1m": "beir-v1.0.0-signal1m",
}


@dataclass
class BenchmarkEntry:
    """Resolved registry entry. All fields are populated from the YAML."""

    key: str
    name: str
    index_name: str
    topics_name: str
    qrels_name: str
    bm25_k1: float
    bm25_b: float
    runfile_template: str = ""
    eval_metrics: list[str] = field(default_factory=list)

    # Optional local files (override Pyserini topics/qrels when set).
    queries_path: Path | None = None
    qrels_path: Path | None = None

    # Hooks for special-cases (e.g. a non-prebuilt local index path).
    is_prebuilt: bool = True

    @property
    def uses_file_queries(self) -> bool:
        return self.queries_path is not None

    @property
    def uses_file_qrels(self) -> bool:
        return self.qrels_path is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "name": self.name,
            "index_name": self.index_name,
            "topics_name": self.topics_name,
            "qrels_name": self.qrels_name,
            "queries_file": str(self.queries_path) if self.queries_path else None,
            "qrels_file": str(self.qrels_path) if self.qrels_path else None,
            "bm25_k1": self.bm25_k1,
            "bm25_b": self.bm25_b,
            "is_prebuilt": self.is_prebuilt,
            "runfile_template": self.runfile_template,
            "eval_metrics": list(self.eval_metrics),
        }


class Registry:
    """Loaded `dataset_registry.yaml` with convenience lookup."""

    def __init__(self, path: str | Path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"dataset registry not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if "datasets" not in data or not isinstance(data["datasets"], dict):
            raise ValueError(
                f"registry at {path} has no `datasets` mapping; got keys: "
                f"{list(data.keys())}"
            )
        self._raw = data
        self._datasets: dict[str, dict[str, Any]] = data["datasets"]
        self._path = path

    @property
    def path(self) -> Path:
        return self._path

    def keys(self) -> list[str]:
        return list(self._datasets.keys())

    def has(self, key: str) -> bool:
        return self._resolve_alias(key) in self._datasets

    def _resolve_alias(self, key: str) -> str:
        if key in self._datasets:
            return key
        if key in _ALIASES and _ALIASES[key] in self._datasets:
            return _ALIASES[key]
        return key

    def get(self, key: str) -> BenchmarkEntry:
        rkey = self._resolve_alias(key)
        if rkey not in self._datasets:
            raise KeyError(
                f"benchmark '{key}' not in registry. Known keys: "
                f"{sorted(self._datasets)}"
            )
        entry = self._datasets[rkey] or {}

        index_block = entry.get("index") or {}
        index_name = index_block.get("name")
        index_path_local = index_block.get("path")
        is_prebuilt = True
        if not index_name:
            if index_path_local:
                # Local index — resolve relative to registry dir
                p = Path(str(index_path_local)).expanduser()
                if not p.is_absolute():
                    p = (self._path.parent / p).resolve()
                index_name = str(p)
                is_prebuilt = False
            else:
                raise ValueError(
                    f"registry entry '{rkey}' is missing index.name or index.path."
                )

        topics_block = entry.get("topics") or {}
        qrels_block = entry.get("qrels") or {}
        topics_name = str(topics_block.get("name") or "")
        qrels_name = str(qrels_block.get("name") or "")

        queries_path = _resolve_registry_file(
            topics_block.get("file"), self._path.parent
        )
        qrels_path = _resolve_registry_file(
            qrels_block.get("file"), self._path.parent
        )

        if not index_name:
            raise ValueError(f"registry entry '{rkey}' is missing index.name.")
        if not queries_path and not topics_name:
            raise ValueError(
                f"registry entry '{rkey}' needs topics.name or topics.file."
            )
        if not qrels_path and not qrels_name:
            raise ValueError(
                f"registry entry '{rkey}' needs qrels.name or qrels.file."
            )

        weights = entry.get("bm25_weights") or {}
        out = entry.get("output") or {}
        return BenchmarkEntry(
            key=rkey,
            name=entry.get("name", rkey),
            index_name=index_name,
            topics_name=topics_name or rkey,
            qrels_name=qrels_name or rkey,
            bm25_k1=float(weights.get("k1", 0.9)),
            bm25_b=float(weights.get("b", 0.4)),
            runfile_template=str(out.get("runfile_template", "")),
            eval_metrics=list(out.get("eval_metrics", [])),
            queries_path=queries_path,
            qrels_path=qrels_path,
            is_prebuilt=is_prebuilt,
        )


def _resolve_registry_file(
    path_value: str | None, registry_dir: Path
) -> Path | None:
    """Resolve a registry ``topics.file`` / ``qrels.file`` path."""
    if not path_value:
        return None
    p = Path(str(path_value)).expanduser()
    if p.is_absolute():
        if not p.is_file():
            raise FileNotFoundError(f"Registry file not found: {p}")
        return p.resolve()
    for base in (registry_dir, registry_dir.parent):
        candidate = (base / p).resolve()
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Registry file '{path_value}' not found relative to {registry_dir} "
        f"or its parent."
    )


# ---------- Pyserini-backed loaders --------------------------------------


def load_pyserini_topics(topics_name: str) -> list[tuple[str, str]]:
    """Return `[(qid, query_text), ...]` using Pyserini's bundled topics.

    Pyserini exposes well-known query collections (TREC DL, BEIR, MSMARCO dev,
    etc.) via `pyserini.search.get_topics`, returning dict[qid -> fields].
    Field name is normally "title"; fall back to "text" for BEIR-style.
    """
    try:
        from pyserini.search import get_topics  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "pyserini is required to load topics by name. "
            "pip install pyserini>=0.22.0"
        ) from e

    raw = get_topics(topics_name)
    if not raw:
        raise RuntimeError(f"pyserini.get_topics returned empty for '{topics_name}'.")

    out: list[tuple[str, str]] = []
    for qid, fields in raw.items():
        if isinstance(fields, dict):
            text = (
                fields.get("title")
                or fields.get("text")
                or fields.get("query")
                or fields.get("description")
                or ""
            )
        else:
            text = str(fields)
        out.append((str(qid), str(text).strip()))
    out.sort(key=lambda t: t[0])
    return out


def dump_pyserini_qrels(qrels_name: str, out_path: str | Path) -> str:
    """Resolve a Pyserini qrels-name to a TREC-format file `trec_eval` can
    read. Returns the file path (creates if missing).
    """
    out_path = Path(out_path)
    if out_path.exists():
        return str(out_path)

    try:
        from pyserini.search import get_qrels  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "pyserini is required to materialise qrels."
        ) from e

    raw = get_qrels(qrels_name)
    if not raw:
        raise RuntimeError(f"pyserini.get_qrels returned empty for '{qrels_name}'.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_lines = 0
    with out_path.open("w", encoding="utf-8") as f:
        # get_qrels -> dict[qid -> dict[doc_id -> relevance]]
        for qid, docs in raw.items():
            if not isinstance(docs, dict):
                continue
            for doc_id, rel in docs.items():
                try:
                    rel_i = int(rel)
                except Exception:
                    rel_i = 0
                f.write(f"{qid} 0 {doc_id} {rel_i}\n")
                n_lines += 1
    logger.info("Wrote %d qrels lines to %s", n_lines, out_path)
    return str(out_path)


def resolve_qrels_path(
    entry: BenchmarkEntry,
    cache_dir: str | Path = ".cache/qrels",
) -> str:
    """Return a TREC qrels file path for ``trec_eval`` (local file or Pyserini)."""
    if entry.qrels_path is not None:
        return str(entry.qrels_path)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / f"{entry.qrels_name}.qrels"
    return dump_pyserini_qrels(entry.qrels_name, out_path)
