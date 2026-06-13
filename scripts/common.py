"""Shared CLI helpers: load config + registry, build SEEK agent + tools."""
from __future__ import annotations

import csv
import json
import logging
import os
import re
import sys
from io import StringIO
from pathlib import Path
from typing import Any

import yaml

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.runner import SEEKRunner  # noqa: E402
from src.llm_client import LLMClient  # noqa: E402
from src.registry import (  # noqa: E402
    BenchmarkEntry,
    Registry,
    load_pyserini_topics,
    resolve_qrels_path,
)
from src.tools.assessor import Judge  # noqa: E402
from src.tools.bm25_searcher import BM25Retriever  # noqa: E402
from src.tools.generator import PseudoPassageGenerator  # noqa: E402
from src.tools.retriever import DenseRetriever  # noqa: E402


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise SystemExit(f"Config at {path} is not a YAML mapping.")
    return cfg


def _resolve_umbrela_template_path() -> str:
    p = _REPO_ROOT / "prompts" / "assessor.yaml"
    if not p.exists():
        raise SystemExit(f"Assessor prompt not found at {p}.")
    return str(p)


def load_registry(cfg: dict[str, Any]) -> Registry:
    reg_cfg = cfg.get("registry") or {}
    path_str = reg_cfg.get("dataset_registry_path") or "dataset_registry.yaml"
    p = Path(path_str)
    if not p.is_absolute():
        p = _REPO_ROOT / p
    return Registry(p)


def resolve_benchmark(
    cfg: dict[str, Any], benchmark: str | None = None
) -> BenchmarkEntry:
    bm = benchmark or cfg.get("eval", {}).get("benchmark")
    if not bm:
        raise SystemExit("No benchmark specified (use --benchmark or eval.benchmark).")
    registry = load_registry(cfg)
    if not registry.has(bm):
        raise SystemExit(
            f"Unknown benchmark '{bm}'. Available registry keys: "
            f"{sorted(registry.keys())}"
        )
    return registry.get(bm)


def model_output_tag(cfg: dict[str, Any]) -> str:
    llm = cfg.get("llm") or {}
    gen = str(llm.get("generator_model", "generator"))
    judge = str(llm.get("judge_model", "judge"))

    def _slug(name: str) -> str:
        s = name.replace("/", "__")
        s = re.sub(r"[^a-zA-Z0-9._-]+", "_", s).strip("._-")
        return s[:96] if s else "model"

    gen_s, judge_s = _slug(gen), _slug(judge)
    if gen_s == judge_s:
        return gen_s
    return f"{gen_s}__judge-{judge_s}"


def eval_path_kwargs(cfg: dict[str, Any], benchmark: str) -> dict[str, str]:
    return {"benchmark": benchmark, "model_tag": model_output_tag(cfg)}


def format_run_output_path(cfg: dict[str, Any], benchmark: str) -> str:
    eval_cfg = cfg.get("eval", {}) or {}
    tpl = eval_cfg.get("run_output", "outputs/runs/{model_tag}/{benchmark}.run")
    return tpl.format(**eval_path_kwargs(cfg, benchmark))


def format_trace_output_dir(cfg: dict[str, Any], benchmark: str) -> Path:
    eval_cfg = cfg.get("eval", {}) or {}
    tpl = eval_cfg.get(
        "trace_output_dir", "outputs/traces/{model_tag}/{benchmark}"
    )
    p = Path(tpl.format(**eval_path_kwargs(cfg, benchmark)))
    if not p.is_absolute():
        p = _REPO_ROOT / p
    return p


def resolve_runs_dir(cfg: dict[str, Any], run_key: str) -> Path:
    p = Path(format_run_output_path(cfg, run_key))
    if not p.is_absolute():
        p = _REPO_ROOT / p
    return p.parent / run_key


# ---------------------------------------------------------------------------
# Retriever instruction resolution
# ---------------------------------------------------------------------------

_RETRIEVER_INSTRUCTIONS_YAML = _REPO_ROOT / "prompts" / "retriever_instructions.yaml"


def _load_instruction_preset(preset_name: str, ret_cfg: dict[str, Any]) -> str | None:
    """Load a named instruction from prompts/retriever_instructions.yaml.

    Returns the instruction string, or None if the preset or file is not found.
    """
    prompts_path = Path(
        ret_cfg.get("instruction_presets_path", str(_RETRIEVER_INSTRUCTIONS_YAML))
    )
    if not prompts_path.is_absolute():
        prompts_path = (_REPO_ROOT / prompts_path).resolve()

    log = logging.getLogger("common")
    if not prompts_path.exists():
        log.warning("retriever_instructions.yaml not found at %s", prompts_path)
        return None

    try:
        with prompts_path.open("r", encoding="utf-8") as f:
            presets = yaml.safe_load(f)
    except Exception as e:
        log.warning("Failed to load instruction presets from %s: %s", prompts_path, e)
        return None

    if not isinstance(presets, dict) or preset_name not in presets:
        log.warning(
            "Instruction preset '%s' not found in %s. Available: %s",
            preset_name,
            prompts_path,
            sorted(presets.keys()) if isinstance(presets, dict) else "?",
        )
        return None

    entry = presets[preset_name]
    instr = entry.get("instruction") if isinstance(entry, dict) else None
    if not instr:
        log.warning("Preset '%s' has no 'instruction' key.", preset_name)
        return None

    desc = entry.get("description", "").strip().replace("\n", " ")
    log.info("Using instruction preset '%s': %s", preset_name, desc[:120])
    return instr


# Map from registry benchmark key fragment to the BRIGHT task name used in
# task configs (e.g. "bright-biology" -> "biology").
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

# Human-readable task names passed to the {task} placeholder in instructions.
_BRIGHT_TASK_DISPLAY: dict[str, str] = {
    "biology": "Biology",
    "earth_science": "Earth Science",
    "economics": "Economics",
    "psychology": "Psychology",
    "robotics": "Robotics",
    "stackoverflow": "Stack Overflow",
    "sustainable_living": "Sustainable Living",
    "pony": "Pony",
    "leetcode": "LeetCode",
    "aops": "AOPS",
    "theoremqa_theorems": "TheoremQA Theorems",
    "theoremqa_questions": "TheoremQA Questions",
}


def _get_retriever_instruction(ret_cfg: dict[str, Any], entry: BenchmarkEntry) -> str:
    """Return the query instruction for the given benchmark.

    Resolution order:
    1. ``instruction_preset`` key — loads a named preset from
       ``prompts/retriever_instructions.yaml`` (overrides everything else).
    2. Per-task JSON config in ``task_configs_dir/<task>.json`` for BRIGHT tasks.
    3. ``default_query_instruction`` config key.
    """
    preset_name = ret_cfg.get("instruction_preset")
    if preset_name:
        preset_instr = _load_instruction_preset(preset_name, ret_cfg)
        if preset_instr is not None:
            task_name = _BRIGHT_TASK_MAP.get(entry.key)
            task_display = _BRIGHT_TASK_DISPLAY.get(task_name, task_name or "")
            return preset_instr.format(task=task_display)

    default_instr = ret_cfg.get(
        "default_query_instruction",
        "<|user|>\nGiven a query, retrieve relevant passages "
        "that help answer the query\n<|embed|>\n",
    )

    task_key = entry.key
    task_name = _BRIGHT_TASK_MAP.get(task_key)
    if task_name is None:
        logger = logging.getLogger("common")
        logger.info(
            "No BRIGHT task mapping for '%s'; using default instruction.", task_key
        )
        return default_instr

    configs_dir = ret_cfg.get("task_configs_dir")
    if configs_dir is None:
        configs_dir = _REPO_ROOT.parent / "evaluation" / "bright" / "configs" / "reasonir"
    else:
        configs_dir = Path(str(configs_dir))
        if not configs_dir.is_absolute():
            configs_dir = (_REPO_ROOT / configs_dir).resolve()

    task_cfg_path = configs_dir / f"{task_name}.json"
    if not task_cfg_path.exists():
        logging.getLogger("common").warning(
            "Task config not found: %s — using default instruction.", task_cfg_path
        )
        return default_instr

    try:
        with task_cfg_path.open("r", encoding="utf-8") as f:
            task_cfg = json.load(f)
    except Exception as e:
        logging.getLogger("common").warning(
            "Failed to read task config %s: %s — using default.", task_cfg_path, e
        )
        return default_instr

    key = "instructions_long" if ret_cfg.get("long_context") else "instructions"
    instr_block = task_cfg.get(key, task_cfg.get("instructions", {}))
    raw = instr_block.get("query", default_instr)
    task_display = _BRIGHT_TASK_DISPLAY.get(task_name, task_name)
    return raw.format(task=task_display)


# ---------------------------------------------------------------------------
# Retriever settings (for pipeline_summary compatibility)
# ---------------------------------------------------------------------------

def _searcher_backend(cfg: dict[str, Any]) -> str:
    backend = str((cfg.get("searcher") or {}).get("backend", "bm25")).lower()
    if backend not in ("bm25", "reasonir"):
        raise SystemExit(
            f"Unknown searcher.backend={backend!r}. Choose 'bm25' or 'reasonir'."
        )
    return backend


def _searcher_settings(
    cfg: dict[str, Any], entry: BenchmarkEntry
) -> tuple[str, bool, float, float]:
    """Resolve (index_path, prebuilt, bm25_k1, bm25_b) using registry + overrides."""
    s = cfg.get("searcher", {}) or {}
    index_override = s.get("index_path_override")
    if index_override:
        index_path = str(index_override)
        prebuilt = bool(s.get("prebuilt", False))
    else:
        index_path = entry.index_name
        prebuilt = entry.is_prebuilt
    bm25_k1 = float(s.get("bm25_k1_override") or entry.bm25_k1)
    bm25_b = float(s.get("bm25_b_override") or entry.bm25_b)
    return index_path, prebuilt, bm25_k1, bm25_b


def resolved_retriever_settings(
    cfg: dict[str, Any], entry: BenchmarkEntry
) -> dict[str, Any]:
    """Return resolved retriever settings for pipeline_summary.md."""
    s_cfg = cfg.get("searcher") or {}
    backend = _searcher_backend(cfg)
    common = {
        "backend": backend,
        "retrieval_depth": int(s_cfg.get("retrieval_depth", 1000)),
        "judge_depth": int(s_cfg.get("judge_depth", 10)),
    }
    if backend == "reasonir":
        ret_cfg = cfg.get("retriever") or {}
        return {
            **common,
            "retriever": "ReasonIR-8B",
            "model_path": ret_cfg.get("model_path", "reasonir/ReasonIR-8B"),
            "embeddings_dir": ret_cfg.get("embeddings_dir", ".cache/seek_embeddings"),
            "instruction_preset": ret_cfg.get("instruction_preset"),
            "max_query_length": int(ret_cfg.get("max_query_length", 512)),
            "max_doc_length": int(ret_cfg.get("max_doc_length", 2048)),
        }
    index_path, prebuilt, bm25_k1, bm25_b = _searcher_settings(cfg, entry)
    return {
        **common,
        "retriever": "BM25",
        "index_path": index_path,
        "prebuilt": prebuilt,
        "bm25_k1": bm25_k1,
        "bm25_b": bm25_b,
    }


# Alias for run_artifacts.py compatibility
resolved_searcher_settings = resolved_retriever_settings


def _build_retriever(cfg: dict[str, Any], entry: BenchmarkEntry):
    search_cfg = cfg.get("searcher", {}) or {}
    backend = _searcher_backend(cfg)
    top_k = int(search_cfg.get("retrieval_depth", 1000))
    snippet_max_tokens = int(search_cfg.get("snippet_max_tokens", 500))
    cache_dir = search_cfg.get("cache_dir")

    if backend == "reasonir":
        ret_cfg = cfg.get("retriever") or {}
        if not ret_cfg:
            raise SystemExit(
                "searcher.backend is 'reasonir' but the retriever: block is missing "
                "from config.yaml. Uncomment it and set instruction_preset."
            )
        embeddings_dir = Path(
            ret_cfg.get("embeddings_dir", ".cache/seek_embeddings")
        )
        if not embeddings_dir.is_absolute():
            embeddings_dir = _REPO_ROOT / embeddings_dir
        task_emb_dir = embeddings_dir / entry.key
        query_instruction = _get_retriever_instruction(ret_cfg, entry)
        logging.getLogger("build_seek_runner").info(
            "Using DenseRetriever: model=%s, embeddings=%s, benchmark=%s",
            ret_cfg.get("model_path", "reasonir/ReasonIR-8B"),
            task_emb_dir,
            entry.key,
        )
        if cache_dir is None:
            cache_dir = str(_REPO_ROOT / ".cache/seek")
        return DenseRetriever(
            embeddings_path=str(task_emb_dir / "embeddings.npy"),
            doc_ids_path=str(task_emb_dir / "doc_ids.json"),
            doc_texts_path=str(task_emb_dir / "doc_texts.jsonl"),
            model_path=ret_cfg.get("model_path", "reasonir/ReasonIR-8B"),
            query_instruction=query_instruction,
            top_k=top_k,
            snippet_max_tokens=snippet_max_tokens,
            cache_dir=cache_dir,
            encode_batch_size=int(ret_cfg.get("encode_batch_size", 1)),
            max_query_length=int(ret_cfg.get("max_query_length", 512)),
        )

    index_path, prebuilt, bm25_k1, bm25_b = _searcher_settings(cfg, entry)
    logging.getLogger("build_seek_runner").info(
        "Using BM25Retriever: index=%s prebuilt=%s bm25_k1=%.3f bm25_b=%.3f (benchmark=%s)",
        index_path, prebuilt, bm25_k1, bm25_b, entry.key,
    )
    return BM25Retriever(
        index_path=index_path,
        prebuilt=prebuilt,
        bm25_k1=bm25_k1,
        bm25_b=bm25_b,
        top_k=top_k,
        snippet_max_tokens=snippet_max_tokens,
        cache_dir=cache_dir if cache_dir is not None else ".cache/bm25",
    )


# ---------------------------------------------------------------------------
# Build SEEK runner
# ---------------------------------------------------------------------------

def build_seek_runner(cfg: dict[str, Any], entry: BenchmarkEntry) -> SEEKRunner:
    llm_cfg = cfg["llm"]
    search_cfg = cfg.get("searcher", {}) or {}
    gen_cfg = cfg.get("generator", {}) or {}
    agent_cfg = cfg.get("agent", {}) or {}
    api_key = os.environ.get(llm_cfg.get("api_key_env", "OPENAI_API_KEY"))
    if not api_key:
        raise SystemExit(
            f"Env var {llm_cfg.get('api_key_env', 'OPENAI_API_KEY')} is not set."
        )

    llm = LLMClient(
        api_key=api_key,
        base_url=llm_cfg.get("base_url"),
        cache_dir=llm_cfg.get("cache_dir", ".cache/llm"),
        max_retries=int(llm_cfg.get("max_retries", 3)),
        extra_body=llm_cfg.get("extra_body"),
    )

    retriever = _build_retriever(cfg, entry)

    # ── Build Judge (UMBRELA assessor) ────────────────────────────────────
    judge = Judge(
        llm=llm,
        model=llm_cfg["judge_model"],
        umbrela_template_path=_resolve_umbrela_template_path(),
        temperature=float(llm_cfg.get("judge_temperature", 0.0)),
        max_tokens=int(llm_cfg.get("judge_max_tokens", 256)),
        num_threads=int(llm_cfg.get("num_threads", 4)),
    )

    # ── Build PseudoPassageGenerator ─────────────────────────────────────
    iter_yaml = gen_cfg.get("iter_prompt_yaml")
    if iter_yaml and not Path(str(iter_yaml)).is_absolute():
        iter_yaml = _REPO_ROOT / iter_yaml

    rf_fallback_yaml = gen_cfg.get("rf_fallback_prompt_yaml")
    if rf_fallback_yaml and not Path(str(rf_fallback_yaml)).is_absolute():
        rf_fallback_yaml = _REPO_ROOT / rf_fallback_yaml

    generator = PseudoPassageGenerator(
        llm=llm,
        model=llm_cfg["generator_model"],
        k_pseudo_refs=int(gen_cfg.get("k_pseudo_refs", 5)),
        alpha=int(gen_cfg.get("alpha_repeat", 5)),
        temperature=float(llm_cfg.get("generator_temperature", 1.0)),
        max_tokens=int(llm_cfg.get("generator_max_tokens", 1024)),
        round1_temperature=float(gen_cfg.get("round1_temperature", 1.0)),
        round1_max_tokens=int(gen_cfg.get("round1_max_tokens", 128)),
        num_threads=int(llm_cfg.get("num_threads", 4)),
        iter_prompt_yaml=iter_yaml,
        iter_mode=gen_cfg.get("iter_mode", "llm"),
        relevance_feedback_max_docs=int(gen_cfg.get("relevance_feedback_max_docs", 5)),
        relevance_feedback_min_score=int(gen_cfg.get("relevance_feedback_min_score", 2)),
        rf_fallback_prompt_yaml=rf_fallback_yaml,
        evidence_max_chars=gen_cfg.get("max_chars"),
        log_prompts=bool(gen_cfg.get("log_prompts", False)),
        skip_round1_generation=bool(gen_cfg.get("skip_round1_generation", False)),
        accumulate_pseudo_refs_from_r2=bool(
            gen_cfg.get("accumulate_pseudo_refs_from_r2", False)
        ),
        api_key=api_key,
        base_url=llm_cfg.get("base_url"),
    )

    query_format = str(agent_cfg.get("query_format", "expanded"))
    logging.getLogger("build_seek_runner").info(
        "Query format for retrieval: %s (backend=%s)",
        query_format,
        _searcher_backend(cfg),
    )

    return SEEKRunner(
        retriever=retriever,
        judge=judge,
        generator=generator,
        max_rounds=int(agent_cfg.get("max_rounds", 5)),
        judge_depth=int(search_cfg.get("judge_depth", 10)),
        termination_all_score_3=bool(agent_cfg.get("termination_all_score_3", True)),
        quality_saturation_min_score=int(
            agent_cfg.get("quality_saturation_min_score", 2)
        ),
        termination_coverage_saturation=bool(
            agent_cfg.get("termination_coverage_saturation", True)
        ),
        saturation_min_new_docs=int(agent_cfg.get("saturation_min_new_docs", 3)),
        saturation_consecutive_rounds=int(
            agent_cfg.get("saturation_consecutive_rounds", 2)
        ),
        evidence_scope=str(agent_cfg.get("evidence_scope", "all")),
        query_format=query_format,
    )


# Alias for backward compatibility
build_seek_agent = build_seek_runner
build_agent = build_seek_runner


# ---------------------------------------------------------------------------
# Query loading helpers
# ---------------------------------------------------------------------------

def load_benchmark_queries(
    cfg: dict[str, Any], entry: BenchmarkEntry
) -> list[tuple[str, str]]:
    if entry.queries_path is not None:
        return load_queries_tsv(entry.queries_path)
    return load_pyserini_topics(entry.topics_name)


def resolve_benchmark_qrels(cfg: dict[str, Any], entry: BenchmarkEntry) -> str:
    cache_dir = (cfg.get("eval") or {}).get("qrels_cache_dir", ".cache/qrels")
    return resolve_qrels_path(entry, cache_dir=cache_dir)


def load_queries_tsv(path: str | Path) -> list[tuple[str, str]]:
    """Load (qid, query_text) rows from a TSV or JSONL file."""
    import json as _json

    path = Path(path)
    if not path.is_file():
        raise SystemExit(f"Queries file not found: {path}")
    raw = path.read_text(encoding="utf-8")
    if not raw.strip():
        return []

    first_line = ""
    for line in raw.splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            first_line = s
            break
    if not first_line:
        return []

    if first_line.startswith("{"):
        id_keys = ("_id", "id", "qid", "query_id", "topic_id")
        text_keys = ("text", "query", "title", "query_text")
        out_json: list[tuple[str, str]] = []
        for line in raw.splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            try:
                obj = _json.loads(s)
            except _json.JSONDecodeError:
                continue
            qid = next((str(obj[k]) for k in id_keys if k in obj and obj[k]), "")
            qtext = next((str(obj[k]) for k in text_keys if k in obj and obj[k]), "")
            if qid and qtext:
                out_json.append((qid, qtext))
        return out_json

    def _norm_cell(s: str) -> str:
        return s.strip().lower().lstrip("\ufeff")

    cells = first_line.split("\t")
    if len(cells) >= 2:
        c0, c1 = _norm_cell(cells[0]), _norm_cell(cells[1])
        id_names = ("qid", "query_id", "topic_id")
        text_names = ("query_text", "query", "title", "text")
        if c0 in id_names and c1 in text_names:
            reader = csv.DictReader(StringIO(raw), delimiter="\t")
            out: list[tuple[str, str]] = []
            for row in reader:
                if not row:
                    continue
                keys = {k.strip().lower(): (v or "").strip() for k, v in row.items() if k}
                qid = next((keys[k] for k in id_names if k in keys and keys[k]), "")
                qtext = next((keys[k] for k in text_names if k in keys and keys[k]), "")
                if qid and qtext:
                    out.append((qid, qtext))
            return out

    out2: list[tuple[str, str]] = []
    for line in raw.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        qid, sep, rest = s.partition("\t")
        if not sep:
            continue
        qid, rest = qid.strip(), rest.strip()
        if qid and rest:
            out2.append((qid, rest))
    return out2


def safe_tag_from_path(path: str | Path) -> str:
    stem = Path(path).stem
    stem = re.sub(r"[^a-zA-Z0-9_-]+", "_", stem).strip("_") or "queries"
    return stem[:80]
