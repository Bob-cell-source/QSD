#!/usr/bin/env bash
set -euo pipefail

# Probe train-only behavior neighbors for LC-SoftSID.
# The current best no-behavior method is kept as a baseline and all outputs go
# under ${DATASET_DIR}/lc_soft_behavior_probe by default.

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET_DIR="${DATASET_DIR:-runs/office}"
SEMANTIC_IDS="${SEMANTIC_IDS:-${DATASET_DIR}/semantic_ids_rq.json}"
OUT_ROOT="${OUT_ROOT:-${DATASET_DIR}/lc_soft_behavior_probe}"
DEVICE="${DEVICE:-cuda}"

EPOCHS="${EPOCHS:-30}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-5}"
BATCH_SIZE="${BATCH_SIZE:-256}"
EVAL_BATCH_EVAL_SIZE="${EVAL_BATCH_EVAL_SIZE:-256}"
MAX_LEN="${MAX_LEN:-50}"
DIM="${DIM:-128}"
NUM_RANDOM_NEG="${NUM_RANDOM_NEG:-100}"
SEED="${SEED:-2026}"
FORCE="${FORCE:-0}"

SCRIPT_START_TS="$(date +%s)"

format_seconds() {
  local total="$1"
  local hours=$((total / 3600))
  local minutes=$(((total % 3600) / 60))
  local seconds=$((total % 60))
  printf "%02d:%02d:%02d" "${hours}" "${minutes}" "${seconds}"
}

COMMON_ARGS=(
  --dataset-dir "${DATASET_DIR}"
  --semantic-ids "${SEMANTIC_IDS}"
  --device "${DEVICE}"
  --epochs "${EPOCHS}"
  --early-stop-patience "${EARLY_STOP_PATIENCE}"
  --batch-size "${BATCH_SIZE}"
  --eval-batch-eval-size "${EVAL_BATCH_EVAL_SIZE}"
  --max-len "${MAX_LEN}"
  --dim "${DIM}"
  --num-hard-neg 0
  --num-random-neg "${NUM_RANDOM_NEG}"
  --lr 0.001
  --weight-decay 0.0001
  --seed "${SEED}"
)

SOFT_BASE_ARGS=(
  --model-variant crsid_soft
  --cr-tail-tau 20
  --cr-residual-scale 1.0
  --cr-soft-top-m 4
  --cr-soft-min-overlap-slots 2
  --cr-soft-min-support 0.05
  --cr-soft-support-eta 2.0
  --cr-soft-hard-token-prior 1.0
  --cr-soft-reliability-floor 0.10
  --cr-soft-max-neighbors 50
  --cr-soft-lift-kappa 0.0
  --cr-alpha-frequency-transform raw
  --sem-weight 1.0
  --dis-weight 0.2
  --div-weight 0.01
)

run_exp() {
  local name="$1"
  shift
  local output_dir="${OUT_ROOT}/${name}"
  local start_ts
  local end_ts
  local elapsed

  if [[ "${FORCE}" != "1" && -f "${output_dir}/test_metrics.json" ]]; then
    echo "Skip existing: ${name}"
    return
  fi

  mkdir -p "${OUT_ROOT}"
  start_ts="$(date +%s)"

  echo "============================================================"
  echo "Running: ${name}"
  echo "Dataset: ${DATASET_DIR}"
  echo "Output:  ${output_dir}"
  echo "Start:   $(date '+%Y-%m-%d %H:%M:%S')"
  echo "============================================================"

  "${PYTHON_BIN}" scripts/train_qsdrec.py \
    "${COMMON_ARGS[@]}" \
    --output-dir "${output_dir}" \
    "$@"

  end_ts="$(date +%s)"
  elapsed=$((end_ts - start_ts))
  echo "Finished: ${name}"
  echo "Elapsed:  $(format_seconds "${elapsed}")"
  echo
}

echo "Dataset dir:  ${DATASET_DIR}"
echo "Semantic IDs: ${SEMANTIC_IDS}"
echo "Output root:  ${OUT_ROOT}"

run_exp "00_soft_no_behavior" \
  "${SOFT_BASE_ARGS[@]}"

run_exp "10_soft_behavior_w025_win5_c2_n50" \
  "${SOFT_BASE_ARGS[@]}" \
  --cr-soft-behavior-weight 0.25 \
  --cr-soft-behavior-window 5 \
  --cr-soft-behavior-min-count 2 \
  --cr-soft-max-behavior-neighbors 50

run_exp "11_soft_behavior_w050_win5_c2_n50" \
  "${SOFT_BASE_ARGS[@]}" \
  --cr-soft-behavior-weight 0.50 \
  --cr-soft-behavior-window 5 \
  --cr-soft-behavior-min-count 2 \
  --cr-soft-max-behavior-neighbors 50

run_exp "12_soft_behavior_w100_win5_c2_n50" \
  "${SOFT_BASE_ARGS[@]}" \
  --cr-soft-behavior-weight 1.00 \
  --cr-soft-behavior-window 5 \
  --cr-soft-behavior-min-count 2 \
  --cr-soft-max-behavior-neighbors 50

# Lower min_count can rescue rarer item-item relations, but may introduce more
# noisy user-history topics. Keep it as a diagnostic variant.
run_exp "13_soft_behavior_w050_win5_c1_n50" \
  "${SOFT_BASE_ARGS[@]}" \
  --cr-soft-behavior-weight 0.50 \
  --cr-soft-behavior-window 5 \
  --cr-soft-behavior-min-count 1 \
  --cr-soft-max-behavior-neighbors 50

"${PYTHON_BIN}" scripts/summarize_experiments.py \
  --root "${OUT_ROOT}" \
  --metric NDCG@10 \
  --top-k 20 \
  --csv "${OUT_ROOT}/summary.csv"

SCRIPT_END_TS="$(date +%s)"
SCRIPT_ELAPSED=$((SCRIPT_END_TS - SCRIPT_START_TS))
echo "============================================================"
echo "Finished LC-SoftSID behavior-neighbor probe"
echo "Summary: ${OUT_ROOT}/summary.csv"
echo "Total elapsed: $(format_seconds "${SCRIPT_ELAPSED}")"
echo "============================================================"
