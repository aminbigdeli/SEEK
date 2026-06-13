"""Batch benchmark runner for SEEK.

Uses ``DenseRetriever`` for dense retrieval each round. After all queries
finish, writes per-mode ``.run`` files and runs ``score.py`` to produce
``scores.csv`` with nDCG@10 via pyserini.

Resume capability: queries whose trace file already exists are skipped.
"""
from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from common import (  # type: ignore
    build_seek_runner,
    eval_path_kwargs,
    format_run_output_path,
    format_trace_output_dir,
    load_benchmark_queries,
    load_config,
    load_queries_tsv,
    model_output_tag,
    resolve_benchmark,
    resolve_runs_dir,
    safe_tag_from_path,
    setup_logging,
)
from run_artifacts import write_run_directory_artifacts  # type: ignore
from src.runner import SEEKRunner, persist_trace  # noqa: E402
from src.fusion import fuse  # noqa: E402
from src.ranking import write_trec_run_fused  # noqa: E402
from src.schemas import JudgeResult, Round, SearchHistory  # noqa: E402

logger = logging.getLogger("seek.run")


def _trace_path(trace_dir: Path, qid: str) -> Path:
    return trace_dir / f"{qid}.trace.json"


def _load_trace(path: Path) -> SearchHistory | None:
    """Best-effort load of an existing trace for resume."""
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None

    hist = SearchHistory(
        query_id=data["query_id"],
        original_query=data["original_query"],
    )
    hist.termination_reason = data.get("termination_reason")
    hist.doc_judge_scores = {k: int(v) for k, v in (data.get("doc_judge_scores") or {}).items()}
    hist.doc_best_score = {k: float(v) for k, v in (data.get("doc_best_score") or {}).items()}
    hist.doc_judge_parse_failed = set(data.get("doc_judge_parse_failed") or [])
    hist.seen_doc_ids = set(hist.doc_judge_scores) | hist.doc_judge_parse_failed

    for rd_data in data.get("rounds", []):
        rd = Round(round_idx=rd_data.get("round_idx", 0))
        rd.retrieved_doc_ids = rd_data.get("retrieved_doc_ids", [])
        rd.new_doc_ids = rd_data.get("new_doc_ids", [])
        rd.n_new_docs_pool = int(rd_data.get("n_new_docs", rd_data.get("n_new_docs_pool", 0)))
        rd.duplicate_count = rd_data.get("duplicate_count", 0)
        rd.retrieval_scores = {
            k: float(v) for k, v in (rd_data.get("retrieval_scores") or {}).items()
        }
        parse_failures = set(rd_data.get("judge_parse_failures", []))
        for d, s in (rd_data.get("judge_scores") or {}).items():
            rd.judge_results[d] = JudgeResult(
                doc_id=d,
                score=int(s),
                parse_failed=(d in parse_failures),
            )
        hist.rounds.append(rd)
        hist.seen_doc_ids.update(rd.retrieved_doc_ids)

    return hist


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run SEEK over a benchmark."
    )
    p.add_argument("--config", default="config.yaml")
    p.add_argument(
        "--benchmark",
        default=None,
        help="Registry key (e.g. bright-biology) or alias.",
    )
    p.add_argument("--num-threads", type=int, default=None)
    p.add_argument(
        "--num-queries",
        type=int,
        default=None,
        help="Optional cap for quick smoke tests.",
    )
    p.add_argument("--log-level", default="INFO")
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="Re-run all queries even if a trace file already exists.",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        metavar="DIR",
        help=(
            "Root directory for all outputs. Sets run files to "
            "DIR/runs/{model_tag}/{benchmark}.run and traces to "
            "DIR/traces/{model_tag}/{benchmark}/. "
            "Overrides both eval.run_output and eval.trace_output_dir."
        ),
    )
    p.add_argument(
        "--run-output",
        default=None,
        help="Override eval.run_output (path of the TREC run file).",
    )
    p.add_argument(
        "--queries-tsv",
        default=None,
        metavar="PATH",
        help="Tab-separated file with qid and query_text columns.",
    )
    p.add_argument(
        "--output-tag",
        default=None,
        help="Suffix for trace dir and run file when using --queries-tsv.",
    )
    return p.parse_args()


def _process_one(
    runner: SEEKRunner,
    qid: str,
    query: str,
    trace_dir: Path,
    resume: bool,
    fusion_cfg: dict | None,
) -> SearchHistory:
    trace_path = _trace_path(trace_dir, qid)
    if resume and trace_path.exists():
        existing = _load_trace(trace_path)
        if existing is not None and existing.doc_judge_scores:
            return existing
    try:
        hist = runner.run(qid, query)
    except Exception as e:
        logger.exception("Runner crashed on qid=%s: %s", qid, e)
        hist = SearchHistory(query_id=qid, original_query=query)
        hist.termination_reason = f"error: {e}"
    persist_trace(hist, trace_path, fusion_cfg=fusion_cfg)
    return hist


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    cfg = load_config(args.config)

    if args.num_threads is not None:
        cfg.setdefault("eval", {})["parallelism"] = args.num_threads
    parallelism = int(cfg.get("eval", {}).get("parallelism", 4))

    entry = resolve_benchmark(cfg, benchmark=args.benchmark)
    if args.queries_tsv:
        queries = load_queries_tsv(args.queries_tsv)
        tag = args.output_tag or safe_tag_from_path(args.queries_tsv)
        run_key = f"{entry.key}_{tag}"
        logger.info(
            "Loaded %d queries from TSV %s (run_key=%s)",
            len(queries), args.queries_tsv, run_key,
        )
    else:
        queries = load_benchmark_queries(cfg, entry)
        run_key = entry.key
        logger.info(
            "Loaded %d queries for benchmark=%s",
            len(queries), entry.key,
        )

    if args.num_queries is not None:
        queries = queries[: args.num_queries]

    eval_cfg = cfg.get("eval", {})
    path_kw = eval_path_kwargs(cfg, run_key)
    if args.output_dir:
        base = Path(args.output_dir)
        run_output = str(base / "runs" / path_kw["model_tag"] / f"{run_key}.run")
        trace_dir = base / "traces" / path_kw["model_tag"] / run_key
    elif args.run_output:
        run_output = args.run_output.format(**path_kw)
        trace_dir = format_trace_output_dir(cfg, run_key)
    else:
        run_output = format_run_output_path(cfg, run_key)
        trace_dir = format_trace_output_dir(cfg, run_key)
    trace_dir.mkdir(parents=True, exist_ok=True)
    Path(run_output).parent.mkdir(parents=True, exist_ok=True)

    runner = build_seek_runner(cfg, entry)
    fusion_cfg = cfg.get("fusion") or {}

    hist_by_qid: dict[str, SearchHistory] = {}
    if parallelism <= 1:
        for qid, q in tqdm(queries, desc="queries"):
            hist_by_qid[qid] = _process_one(
                runner, qid, q, trace_dir, resume=not args.no_resume,
                fusion_cfg=fusion_cfg or None,
            )
    else:
        with ThreadPoolExecutor(max_workers=parallelism) as ex:
            futures = {
                ex.submit(
                    _process_one, runner, qid, q, trace_dir, not args.no_resume,
                    fusion_cfg or None,
                ): (qid, q)
                for qid, q in queries
            }
            for fut in tqdm(as_completed(futures), total=len(futures), desc="queries"):
                qid, q = futures[fut]
                try:
                    hist_by_qid[qid] = fut.result()
                except Exception as e:
                    logger.exception("Thread failed on qid=%s: %s", qid, e)
                    hist = SearchHistory(query_id=qid, original_query=q)
                    hist.termination_reason = f"error: {e}"
                    hist_by_qid[qid] = hist

    histories = [hist_by_qid[qid] for qid, _ in queries]
    run_tag = eval_cfg.get("run_tag", "seek")
    modes = fusion_cfg.get("modes_to_run", [])
    primary = fusion_cfg.get("primary_mode", "judge_only_top_10")

    runs_dir = resolve_runs_dir(cfg, run_key)
    runs_dir.mkdir(parents=True, exist_ok=True)

    if modes:
        for mode in modes:
            mode_path = runs_dir / f"{mode}.run"
            n = write_trec_run_fused(histories, mode_path, mode, fusion_cfg, run_tag)
            logger.info("Wrote %d lines to %s", n, mode_path)

    n_lines = 0
    with open(run_output, "w", encoding="utf-8") as f:
        for hist in histories:
            ranked = fuse(hist, primary, fusion_cfg)
            for doc_id, rank, score in ranked:
                f.write(
                    f"{hist.query_id} Q0 {doc_id} {rank} {score:.4f} {run_tag}\n"
                )
                n_lines += 1
    logger.info("Wrote %d lines to %s (primary=%s)", n_lines, run_output, primary)

    if modes:
        from score import run_scoring  # type: ignore

        run_scoring(
            cfg,
            benchmark=args.benchmark,
            run_key=run_key,
            config_path=args.config,
            num_queries=len(queries),
        )
    else:
        write_run_directory_artifacts(
            cfg,
            entry,
            runs_dir,
            config_path=args.config,
            run_key=run_key,
            num_queries=len(queries),
            trace_dir=trace_dir,
            legacy_run_file=Path(run_output),
            run_tag=run_tag,
        )


if __name__ == "__main__":
    main()
