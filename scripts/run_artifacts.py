"""Write per-benchmark run directory artifacts (summary + evaluation metadata).

Outputs under ``outputs/runs/{run_key}/``:

- ``pipeline_summary.md`` — human-readable hyper-parameters and resolved settings
- ``evaluation_metadata.json`` — machine-readable benchmark, config, and metrics
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import (  # type: ignore
    _REPO_ROOT,
    resolved_retriever_settings,
)

SCHEMA_VERSION = "1"

_TREC_EVAL_METRICS = {
    "MAP": {"metric": "map", "flags": ["-l", "2"]},
    "nDCG@10": {"metric": "ndcg_cut.10", "flags": []},
    "Recall@1000": {"metric": "recall.1000", "flags": ["-l", "2"]},
}


def _git_revision() -> str | None:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if res.returncode == 0:
            return res.stdout.strip() or None
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


def _resolve_config_path(config_path: str | Path) -> Path:
    p = Path(config_path)
    if not p.is_absolute():
        p = _REPO_ROOT / p
    return p.resolve()


def _effective_generator_settings(gen_cfg: dict[str, Any]) -> dict[str, Any]:
    """Generator / iter settings including code defaults for omitted keys."""
    return {
        "k_pseudo_refs": int(gen_cfg.get("k_pseudo_refs", 5)),
        "alpha_repeat": int(gen_cfg.get("alpha_repeat", 5)),
        "round1_temperature": float(gen_cfg.get("round1_temperature", 1.0)),
        "round1_max_tokens": int(gen_cfg.get("round1_max_tokens", 128)),
        "iter_prompt_yaml": gen_cfg.get("iter_prompt_yaml"),
        "iter_mode": gen_cfg.get("iter_mode", "llm"),
        "relevance_feedback_max_docs": int(
            gen_cfg.get("relevance_feedback_max_docs", 5)
        ),
        "relevance_feedback_min_score": int(
            gen_cfg.get("relevance_feedback_min_score", 2)
        ),
        "rf_fallback_prompt_yaml": gen_cfg.get("rf_fallback_prompt_yaml"),
    }


def build_hyperparameters(cfg: dict[str, Any], entry: Any) -> dict[str, Any]:
    """Structured hyper-parameters from config + resolved registry values."""
    llm = cfg.get("llm") or {}
    searcher = cfg.get("searcher") or {}
    retriever_cfg = cfg.get("retriever") or {}
    generator = cfg.get("generator") or {}
    agent = cfg.get("agent") or {}
    fusion = cfg.get("fusion") or {}
    eval_cfg = cfg.get("eval") or {}

    return {
        "llm": {
            "provider": llm.get("provider"),
            "base_url": llm.get("base_url"),
            "generator_model": llm.get("generator_model"),
            "judge_model": llm.get("judge_model"),
            "fallback_model": llm.get("fallback_model"),
            "generator_temperature": llm.get("generator_temperature"),
            "judge_temperature": llm.get("judge_temperature"),
            "generator_max_tokens": llm.get("generator_max_tokens"),
            "judge_max_tokens": llm.get("judge_max_tokens"),
            "max_retries": llm.get("max_retries"),
            "cache_dir": llm.get("cache_dir"),
            "num_threads": llm.get("num_threads"),
        },
        "searcher": dict(searcher),
        "retriever": {
            **retriever_cfg,
            **resolved_retriever_settings(cfg, entry),
        },
        "generator": _effective_generator_settings(generator),
        "agent": dict(agent),
        "fusion": dict(fusion),
        "eval": dict(eval_cfg),
        "registry": dict(cfg.get("registry") or {}),
    }


def _format_yaml_block(data: dict[str, Any], indent: int = 0) -> list[str]:
    """Simple nested dict/list formatter for markdown (not full YAML dump)."""
    lines: list[str] = []
    prefix = "  " * indent
    for key, value in data.items():
        if isinstance(value, dict):
            lines.append(f"{prefix}- **{key}**:")
            lines.extend(_format_yaml_block(value, indent + 1))
        elif isinstance(value, list):
            lines.append(f"{prefix}- **{key}**: {value!r}")
        else:
            lines.append(f"{prefix}- **{key}**: `{value}`")
    return lines


def write_pipeline_summary(
    runs_dir: Path,
    *,
    run_key: str,
    entry: Any,
    hyperparameters: dict[str, Any],
    config_path: Path,
    num_queries: int | None = None,
    trace_dir: Path | None = None,
) -> Path:
    """Write ``pipeline_summary.md`` and return its path."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    out_path = runs_dir / "pipeline_summary.md"

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        "# Pipeline summary",
        "",
        f"- **Run key**: `{run_key}`",
        f"- **Benchmark**: {entry.name} (`{entry.key}`)",
        f"- **Config file**: `{config_path}`",
        f"- **Generated**: {now}",
    ]
    if num_queries is not None:
        lines.append(f"- **Queries**: {num_queries}")
    if trace_dir is not None:
        lines.append(f"- **Traces**: `{trace_dir}`")
    lines.extend([
        "",
        "## Benchmark (registry)",
        "",
        f"- **Topics**: `{entry.queries_path or entry.topics_name}`",
        f"- **Qrels**: `{entry.qrels_path or entry.qrels_name}`",
        f"- **Index (registry)**: `{entry.index_name}`",
        "",
        "## Hyper-parameters",
        "",
    ])

    for section, params in hyperparameters.items():
        if not params:
            continue
        lines.append(f"### {section}")
        lines.append("")
        if isinstance(params, dict):
            lines.extend(_format_yaml_block(params))
        else:
            lines.append(f"- `{params}`")
        lines.append("")

    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return out_path


def build_evaluation_metadata(
    *,
    run_key: str,
    entry: Any,
    hyperparameters: dict[str, Any],
    config_path: Path,
    runs_dir: Path,
    scores_rows: list[dict[str, Any]] | None = None,
    num_queries: int | None = None,
    trace_dir: Path | None = None,
    legacy_run_file: Path | None = None,
    run_tag: str | None = None,
) -> dict[str, Any]:
    fusion = hyperparameters.get("fusion") or {}
    eval_cfg = hyperparameters.get("eval") or {}
    primary_mode = fusion.get("primary_mode")
    modes = fusion.get("modes_to_run") or []

    run_files = sorted(p.name for p in runs_dir.glob("*.run")) if runs_dir.is_dir() else []

    meta: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repository_root": str(_REPO_ROOT),
        "git_revision": _git_revision(),
        "benchmark": entry.to_dict(),
        "run_key": run_key,
        "config": {
            "path": str(config_path),
            "hyperparameters": hyperparameters,
        },
        "paths": {
            "runs_dir": str(runs_dir.resolve()),
            "trace_dir": str(trace_dir.resolve()) if trace_dir else None,
            "legacy_run_file": str(legacy_run_file.resolve()) if legacy_run_file else None,
            "pipeline_summary": "pipeline_summary.md",
            "scores_csv": "scores.csv" if scores_rows else None,
        },
        "run": {
            "num_queries": num_queries,
            "run_tag": run_tag,
            "fusion_modes": modes,
            "primary_mode": primary_mode,
            "run_files": run_files,
        },
        "evaluation": None,
        "artifacts": {
            "pipeline_summary": "pipeline_summary.md",
            "evaluation_metadata": "evaluation_metadata.json",
        },
    }

    if scores_rows:
        meta["evaluation"] = {
            "qrels": str(entry.qrels_path) if entry.qrels_path else entry.qrels_name,
            "metrics": _TREC_EVAL_METRICS,
            "scores_by_mode": scores_rows,
            "best_ndcg_at_10": max(
                scores_rows,
                key=lambda r: float(r.get("nDCG@10", 0)),
            ).get("mode"),
        }
        meta["artifacts"]["scores_csv"] = "scores.csv"

    registry_metrics = entry.eval_metrics or None
    if registry_metrics:
        meta["evaluation"] = meta["evaluation"] or {}
        meta["evaluation"]["registry_default_metrics"] = registry_metrics

    if eval_cfg.get("metrics"):
        meta["evaluation"] = meta["evaluation"] or {}
        meta["evaluation"]["config_metrics_override"] = eval_cfg["metrics"]

    return meta


def write_evaluation_metadata(runs_dir: Path, metadata: dict[str, Any]) -> Path:
    out_path = runs_dir / "evaluation_metadata.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return out_path


def write_run_directory_artifacts(
    cfg: dict[str, Any],
    entry: Any,
    runs_dir: Path,
    *,
    config_path: str | Path = "config.yaml",
    run_key: str,
    scores_rows: list[dict[str, Any]] | None = None,
    num_queries: int | None = None,
    trace_dir: Path | None = None,
    legacy_run_file: Path | None = None,
    run_tag: str | None = None,
) -> tuple[Path, Path]:
    """Write ``pipeline_summary.md`` and ``evaluation_metadata.json``."""
    config_path = _resolve_config_path(config_path)
    hyperparameters = build_hyperparameters(cfg, entry)
    eval_cfg = cfg.get("eval") or {}
    run_tag = run_tag or eval_cfg.get("run_tag")

    summary_path = write_pipeline_summary(
        runs_dir,
        run_key=run_key,
        entry=entry,
        hyperparameters=hyperparameters,
        config_path=config_path,
        num_queries=num_queries,
        trace_dir=trace_dir,
    )

    metadata = build_evaluation_metadata(
        run_key=run_key,
        entry=entry,
        hyperparameters=hyperparameters,
        config_path=config_path,
        runs_dir=runs_dir,
        scores_rows=scores_rows,
        num_queries=num_queries,
        trace_dir=trace_dir,
        legacy_run_file=legacy_run_file,
        run_tag=run_tag,
    )
    metadata_path = write_evaluation_metadata(runs_dir, metadata)
    return summary_path, metadata_path
