#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/voice/bin/python}"
DATASET_DIR="${DATASET_DIR:-runs/office}"
SEMANTIC_IDS="${SEMANTIC_IDS:-${DATASET_DIR}/semantic_ids_rq.json}"
OUT_ROOT="${OUT_ROOT:-${DATASET_DIR}/crsid_soft_probe}"
DEVICE="${DEVICE:-cuda}"

EPOCHS="${EPOCHS:-30}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-5}"
BATCH_SIZE="${BATCH_SIZE:-256}"
EVAL_BATCH_EVAL_SIZE="${EVAL_BATCH_EVAL_SIZE:-1024}"
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

# Baselines kept in the same output root for a direct summary table.
run_exp "00_sasrec_id_only" \
  --model-variant qsdrec \
  --num-interests 1 \
  --sem-weight 0 \
  --dis-weight 0 \
  --div-weight 0

run_exp "01_qsdrec_semantic_score" \
  --model-variant qsdrec \
  --num-interests 4 \
  --sem-weight 0.10 \
  --dis-weight 0 \
  --div-weight 0

run_exp "10_crsid_hard_tau20_s10" \
  --model-variant crsid \
  --cr-tail-tau 20 \
  --cr-residual-scale 1.0

# Soft SID without local pruning. This isolates whether multi-candidate SID alone helps.
run_exp "20_crsid_soft_m4_no_prune_rel1" \
  --model-variant crsid_soft \
  --cr-tail-tau 20 \
  --cr-residual-scale 1.0 \
  --cr-soft-top-m 4 \
  --cr-soft-min-overlap-slots 2 \
  --cr-soft-min-support 0.0 \
  --cr-soft-support-eta 1.0 \
  --cr-soft-hard-token-prior 1.0 \
  --cr-soft-reliability-floor 1.0 \
  --cr-soft-max-neighbors 50

# Local-consistent soft SID. This is the main candidate.
run_exp "21_crsid_soft_m4_s005_rel010" \
  --model-variant crsid_soft \
  --cr-tail-tau 20 \
  --cr-residual-scale 1.0 \
  --cr-soft-top-m 4 \
  --cr-soft-min-overlap-slots 2 \
  --cr-soft-min-support 0.05 \
  --cr-soft-support-eta 1.0 \
  --cr-soft-hard-token-prior 1.0 \
  --cr-soft-reliability-floor 0.10 \
  --cr-soft-max-neighbors 50

# Slightly stricter local consistency.
run_exp "22_crsid_soft_m4_s010_rel010" \
  --model-variant crsid_soft \
  --cr-tail-tau 20 \
  --cr-residual-scale 1.0 \
  --cr-soft-top-m 4 \
  --cr-soft-min-overlap-slots 2 \
  --cr-soft-min-support 0.10 \
  --cr-soft-support-eta 1.0 \
  --cr-soft-hard-token-prior 1.0 \
  --cr-soft-reliability-floor 0.10 \
  --cr-soft-max-neighbors 50

# Larger candidate set. If this hurts, over-sharing is still entering through soft candidates.
run_exp "23_crsid_soft_m8_s005_rel010" \
  --model-variant crsid_soft \
  --cr-tail-tau 20 \
  --cr-residual-scale 1.0 \
  --cr-soft-top-m 8 \
  --cr-soft-min-overlap-slots 2 \
  --cr-soft-min-support 0.05 \
  --cr-soft-support-eta 1.0 \
  --cr-soft-hard-token-prior 1.0 \
  --cr-soft-reliability-floor 0.10 \
  --cr-soft-max-neighbors 50

# Same soft SID as the main candidate, but disable reliability-calibrated alpha
# by forcing reliability to 1.0. This isolates the effect of reliability alpha.
run_exp "24_crsid_soft_m4_s005_rel1" \
  --model-variant crsid_soft \
  --cr-tail-tau 20 \
  --cr-residual-scale 1.0 \
  --cr-soft-top-m 4 \
  --cr-soft-min-overlap-slots 2 \
  --cr-soft-min-support 0.05 \
  --cr-soft-support-eta 1.0 \
  --cr-soft-hard-token-prior 1.0 \
  --cr-soft-reliability-floor 1.0 \
  --cr-soft-max-neighbors 50

"${PYTHON_BIN}" scripts/summarize_experiments.py \
  --root "${OUT_ROOT}" \
  --metric NDCG@10 \
  --top-k 30 \
  --csv "${OUT_ROOT}/summary.csv"

SCRIPT_END_TS="$(date +%s)"
SCRIPT_ELAPSED=$((SCRIPT_END_TS - SCRIPT_START_TS))
echo "============================================================"
echo "Finished CRSID soft SID experiments"
echo "Summary: ${OUT_ROOT}/summary.csv"
echo "Total elapsed: $(format_seconds "${SCRIPT_ELAPSED}")"
echo "============================================================"
