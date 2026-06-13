"""Write fusion run files + scores.csv from existing trace files.

Use this when traces already exist (e.g. after a prior run) and you
only need to (re)generate the TREC run files and nDCG@10 scores —
without reloading the dense retrieval model on the GPU.

Usage:
  python scripts/write_runs.py --benchmark bright-biology
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from common import (  # type: ignore
    format_trace_output_dir,
    load_config,
    resolve_benchmark,
    resolve_runs_dir,
    setup_logging,
)
from run import _load_trace  # type: ignore
from score import run_scoring  # type: ignore
from src.ranking import write_trec_run_fused  # noqa: E402
from src.schemas import SearchHistory  # noqa: E402

logger = logging.getLogger("seek.write_runs")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Write fusion run files + scores.csv from existing traces."
    )
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--benchmark", required=True, help="Registry key (e.g. bright-biology).")
    p.add_argument("--run-key", default=None, help="Override run directory key.")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    cfg = load_config(args.config)

    entry = resolve_benchmark(cfg, benchmark=args.benchmark)
    run_key = args.run_key or entry.key
    trace_dir = format_trace_output_dir(cfg, run_key)
    if not trace_dir.is_dir():
        raise SystemExit(f"Trace directory not found: {trace_dir}")

    trace_files = sorted(trace_dir.glob("*.trace.json"))
    if not trace_files:
        raise SystemExit(f"No trace files in {trace_dir}")

    histories: list[SearchHistory] = []
    for tf in trace_files:
        hist = _load_trace(tf)
        if hist is not None:
            histories.append(hist)

    if not histories:
        raise SystemExit("Could not load any traces.")

    fusion_cfg = cfg.get("fusion") or {}
    modes = fusion_cfg.get("modes_to_run", [])
    if not modes:
        raise SystemExit("No fusion.modes_to_run in config.yaml.")

    eval_cfg = cfg.get("eval", {})
    run_tag = eval_cfg.get("run_tag", "seek")
    runs_dir = resolve_runs_dir(cfg, run_key)
    runs_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loaded %d traces from %s", len(histories), trace_dir)
    for mode in modes:
        mode_path = runs_dir / f"{mode}.run"
        n = write_trec_run_fused(histories, mode_path, mode, fusion_cfg, run_tag)
        logger.info("Wrote %d lines to %s", n, mode_path)

    run_scoring(
        cfg,
        benchmark=args.benchmark,
        run_key=run_key,
        config_path=args.config,
        num_queries=len(histories),
    )


if __name__ == "__main__":
    main()
