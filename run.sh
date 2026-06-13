#!/usr/bin/env bash
# run.sh — run SEEK for one or more BRIGHT datasets.
#
# Default datasets (no args): bright-earth-science  bright-economics  bright-psychology
#
# Usage:
#   ./run.sh                                    # runs the three defaults
#   ./run.sh bright-biology bright-robotics     # override with any list
#
# Options (env vars, set before calling):
#   CONFIG        config file path           (default: config.yaml)
#   LOG_LEVEL     INFO | DEBUG | WARNING     (default: INFO)
#   NO_RESUME     set to 1 to re-run all queries, even if traces exist
#   NUM_QUERIES   integer cap for quick smoke tests (unset = all)
#   OUTPUT_DIR    root dir for runs/ and traces/ (default: outputs/ inside repo)
#
# Examples:
#   ./run.sh
#   NO_RESUME=1 ./run.sh
#   NUM_QUERIES=5 ./run.sh bright-economics
#   OUTPUT_DIR=/scratch/my_experiment ./run.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"
RUN_SCRIPT="$SCRIPT_DIR/scripts/run.py"

CONFIG="${CONFIG:-config.yaml}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
NO_RESUME="${NO_RESUME:-0}"
NUM_QUERIES="${NUM_QUERIES:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"

# ── Default datasets ──────────────────────────────────────────────────────────
DEFAULT_DATASETS=(
    bright-earth-science
    bright-economics
    bright-psychology
)

if [[ ! -f "$SCRIPT_DIR/$CONFIG" && ! -f "$CONFIG" ]]; then
    echo "ERROR: config file not found: $CONFIG"
    exit 1
fi

# ── Build argument list ───────────────────────────────────────────────────────
EXTRA_ARGS=()
[[ "$NO_RESUME" == "1" ]] && EXTRA_ARGS+=("--no-resume")
[[ -n "$NUM_QUERIES" ]]   && EXTRA_ARGS+=("--num-queries" "$NUM_QUERIES")
[[ -n "$OUTPUT_DIR" ]]    && EXTRA_ARGS+=("--output-dir" "$OUTPUT_DIR")

# ── Run ───────────────────────────────────────────────────────────────────────
if [[ $# -eq 0 ]]; then
    DATASETS=("${DEFAULT_DATASETS[@]}")
else
    DATASETS=("$@")
fi
TOTAL=${#DATASETS[@]}
FAILED=()

echo "========================================"
echo "  SEEK — BRIGHT evaluation"
echo "  config    : $CONFIG"
echo "  output_dir: ${OUTPUT_DIR:-<default: outputs/ in repo>}"
echo "  datasets  : ${DATASETS[*]}"
echo "========================================"

cd "$SCRIPT_DIR"

for i in "${!DATASETS[@]}"; do
    BENCHMARK="${DATASETS[$i]}"
    IDX=$((i + 1))

    echo ""
    echo "[$IDX/$TOTAL] -- $BENCHMARK -------------------------"
    echo "  Starting run.py ..."

    if "$PYTHON" "$RUN_SCRIPT" \
            --config "$CONFIG" \
            --benchmark "$BENCHMARK" \
            --log-level "$LOG_LEVEL" \
            "${EXTRA_ARGS[@]}"; then
        echo "  OK $BENCHMARK done"
    else
        echo "  FAILED $BENCHMARK (exit $?)"
        FAILED+=("$BENCHMARK")
    fi
done

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "  Summary: $((TOTAL - ${#FAILED[@]})) / $TOTAL succeeded"
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "  Failed:"
    for ds in "${FAILED[@]}"; do
        echo "    - $ds"
    done
    exit 1
fi
echo "========================================"
