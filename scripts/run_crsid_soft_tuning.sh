#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/voice/bin/python}"
DATASET_DIR="${DATASET_DIR:-runs/beauty}"
SEMANTIC_IDS="${SEMANTIC_IDS:-${DATASET_DIR}/semantic_ids_rq.json}"
OUT_ROOT="${OUT_ROOT:-${DATASET_DIR}/crsid_soft_tuning}"
DEVICE="${DEVICE:-cuda}"

EPOCHS="${EPOCHS:-100}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-10}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
EVAL_BATCH_EVAL_SIZE="${EVAL_BATCH_EVAL_SIZE:-4096}"
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
  --seed "${SEED}"
  --model-variant crsid_soft
  --cr-tail-tau 20
  --cr-residual-scale 1.0
  --cr-soft-min-overlap-slots 2
  --cr-soft-min-support 0.05
  --cr-soft-reliability-floor 0.10
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

# More conservative candidate set than m=4. Tests whether reduced soft sharing is cleaner.
run_exp "25_crsid_soft_m3_s005_prior1_eta1_n50" \
  --cr-soft-top-m 3 \
  --cr-soft-hard-token-prior 1.0 \
  --cr-soft-support-eta 1.0 \
  --cr-soft-max-neighbors 50

# Keep m=4 but strengthen the original hard token. Tests whether soft SID is
# helping only when it does not dilute the hard assignment too much.
run_exp "26_crsid_soft_m4_s005_prior2_eta1_n50" \
  --cr-soft-top-m 4 \
  --cr-soft-hard-token-prior 2.0 \
  --cr-soft-support-eta 1.0 \
  --cr-soft-max-neighbors 50

# Sharper support weighting. Tokens with stronger local support dominate more.
run_exp "27_crsid_soft_m4_s005_prior1_eta2_n50" \
  --cr-soft-top-m 4 \
  --cr-soft-hard-token-prior 1.0 \
  --cr-soft-support-eta 2.0 \
  --cr-soft-max-neighbors 50

# Smaller local neighborhood. Tests whether broad neighborhoods introduce
# generic office/beauty-topic noise.
run_exp "28_crsid_soft_m4_s005_prior1_eta1_n20" \
  --cr-soft-top-m 4 \
  --cr-soft-hard-token-prior 1.0 \
  --cr-soft-support-eta 1.0 \
  --cr-soft-max-neighbors 20

"${PYTHON_BIN}" scripts/summarize_experiments.py \
  --root "${OUT_ROOT}" \
  --metric NDCG@10 \
  --top-k 20 \
  --csv "${OUT_ROOT}/summary.csv"

SCRIPT_END_TS="$(date +%s)"
SCRIPT_ELAPSED=$((SCRIPT_END_TS - SCRIPT_START_TS))
echo "============================================================"
echo "Finished CRSID soft SID tuning"
echo "Summary: ${OUT_ROOT}/summary.csv"
echo "Total elapsed: $(format_seconds "${SCRIPT_ELAPSED}")"
echo "============================================================"
