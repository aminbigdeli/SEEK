"""Score all fusion-mode run files for a benchmark and write a CSV summary.

Runs after ``run.py`` has produced per-mode .run files under
``outputs/runs/{benchmark}/``. Uses trec_eval for MAP, nDCG@10, and
Recall@1000.

The CSV is written next to the run files:
  outputs/runs/{benchmark}/scores.csv

Can also be invoked standalone:
  python scripts/score.py --benchmark bright-biology
"""
from __future__ import annotations

import argparse
import csv
import logging
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from common import (  # type: ignore
    format_run_output_path,
    format_trace_output_dir,
    load_config,
    resolve_benchmark,
    resolve_benchmark_qrels,
    resolve_runs_dir,
    setup_logging,
)
from run_artifacts import write_run_directory_artifacts  # type: ignore

logger = logging.getLogger("seek.score")

_METRIC_RE = re.compile(r"^\s*(\S+)\s+all\s+([\d.]+)\s*$")


def _trec_eval(
    qrels: str, run_file: str, metric: str, extra_flags: list[str] | None = None,
) -> float:
    """Run a single trec_eval metric and return the 'all' value."""
    cmd = [sys.executable, "-m", "pyserini.eval.trec_eval", "-c"]
    if extra_flags:
        cmd.extend(extra_flags)
    cmd.extend(["-m", metric, qrels, run_file])
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        logger.warning("trec_eval failed on %s: %s", run_file, res.stderr[:200])
        return 0.0
    for line in (res.stdout or "").splitlines():
        m = _METRIC_RE.match(line)
        if m:
            return float(m.group(2))
    return 0.0


def score_all_modes(
    runs_dir: Path, qrels_path: str, metrics: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Score every .run file in *runs_dir* and return rows for CSV."""
    run_files = sorted(runs_dir.glob("*.run"))
    if not run_files:
        logger.warning("No .run files found in %s", runs_dir)
        return []

    metric_specs: list[tuple[str, str, list[str]]] = []
    if metrics:
        for m in metrics:
            if m == "map":
                metric_specs.append(("MAP", "map", ["-l", "2"]))
            elif m in ("ndcg_cut.10", "ndcg@10"):
                metric_specs.append(("nDCG@10", "ndcg_cut.10", []))
            elif m == "recall.1000":
                metric_specs.append(("Recall@1000", "recall.1000", ["-l", "2"]))
            elif m == "recall.100":
                metric_specs.append(("Recall@100", "recall.100", []))
            elif m == "recip_rank":
                metric_specs.append(("MRR", "recip_rank", []))
    if not metric_specs:
        metric_specs = [
            ("MAP", "map", ["-l", "2"]),
            ("nDCG@10", "ndcg_cut.10", []),
            ("Recall@1000", "recall.1000", ["-l", "2"]),
        ]

    rows: list[dict[str, Any]] = []
    for rf in run_files:
        mode = rf.stem
        logger.info("Scoring %s ...", mode)
        row: dict[str, Any] = {"mode": mode}
        parts = [f"{mode:35s}"]
        for col, metric, flags in metric_specs:
            val = _trec_eval(qrels_path, str(rf), metric, flags)
            row[col] = f"{val:.4f}"
            parts.append(f"{col}={val:.4f}")
        rows.append(row)
        print("  " + "  ".join(parts))

    return rows


def write_scores_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {out_path}")


def run_scoring(
    cfg: dict[str, Any],
    benchmark: str | None = None,
    *,
    run_key: str | None = None,
    config_path: str | Path = "config.yaml",
    num_queries: int | None = None,
) -> Path | None:
    """Entry point usable from run.py or standalone."""
    entry = resolve_benchmark(cfg, benchmark=benchmark)
    rk = run_key or entry.key
    runs_dir = resolve_runs_dir(cfg, rk)

    if not runs_dir.is_dir():
        logger.warning("Runs directory not found: %s", runs_dir)
        return None

    eval_cfg = cfg.get("eval", {})
    trace_dir = format_trace_output_dir(cfg, rk)
    legacy_run = Path(format_run_output_path(cfg, rk))

    qrels_path = resolve_benchmark_qrels(cfg, entry)
    print(f"=== Scoring modes for {entry.name} ({entry.key}) ===")
    if entry.uses_file_qrels:
        print(f"  qrels_file: {qrels_path}")
    else:
        print(f"  qrels: {entry.qrels_name} -> {qrels_path}")
    print(f"  runs_dir: {runs_dir}\n")

    rows = score_all_modes(runs_dir, qrels_path, entry.eval_metrics)
    csv_path = runs_dir / "scores.csv"
    write_scores_csv(rows, csv_path)

    n_q = num_queries
    if n_q is None and trace_dir.is_dir():
        n_q = len(list(trace_dir.glob("*.trace.json")))

    summary_path, meta_path = write_run_directory_artifacts(
        cfg,
        entry,
        runs_dir,
        config_path=config_path,
        run_key=rk,
        scores_rows=rows,
        num_queries=n_q,
        trace_dir=trace_dir if trace_dir.is_dir() else None,
        legacy_run_file=legacy_run if legacy_run.is_file() else None,
        run_tag=eval_cfg.get("run_tag"),
    )
    logger.info("Wrote %s and %s", summary_path, meta_path)
    return csv_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score all fusion-mode run files for a benchmark.")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--benchmark", default=None, help="Registry key or alias.")
    p.add_argument(
        "--run-key",
        default=None,
        help="Run directory name under outputs/runs/ (default: benchmark registry key).",
    )
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    cfg = load_config(args.config)
    run_scoring(
        cfg,
        benchmark=args.benchmark,
        run_key=args.run_key,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
