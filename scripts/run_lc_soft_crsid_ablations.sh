#!/usr/bin/env bash
set -euo pipefail

# Paper ablations for LC-Soft CRSID.
#
# Default target is Toys/Games. Override DATASET_DIR / SEMANTIC_IDS / OUT_ROOT
# to reuse this script on Office, Beauty, Baby, or Sports.

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET_DIR="${DATASET_DIR:-runs/toys_games}"
SEMANTIC_IDS="${SEMANTIC_IDS:-${DATASET_DIR}/semantic_ids_rq.json}"
OUT_ROOT="${OUT_ROOT:-${DATASET_DIR}/lc_soft_crsid_ablation}"
DEVICE="${DEVICE:-cuda}"

EPOCHS="${EPOCHS:-100}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-10}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
EVAL_BATCH_EVAL_SIZE="${EVAL_BATCH_EVAL_SIZE:-2048}"
MAX_LEN="${MAX_LEN:-50}"
DIM="${DIM:-128}"
NUM_RANDOM_NEG="${NUM_RANDOM_NEG:-100}"
SEED="${SEED:-2026}"
FORCE="${FORCE:-0}"

# Set RUN_HEAVY=0 on very large datasets to keep only the essential ablations.
RUN_HEAVY="${RUN_HEAVY:-1}"

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

SOFT_MAIN_ARGS=(
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
echo "RUN_HEAVY:    ${RUN_HEAVY}"

# ---------------------------------------------------------------------------
# A. Overall comparisons: prove the final representation is better than pure
# ID, score-level semantic enhancement, and hard Semantic ID.
# ---------------------------------------------------------------------------

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

run_exp "10_hard_crsid_tau20" \
  --model-variant crsid \
  --cr-tail-tau 20 \
  --cr-residual-scale 1.0 \
  --sem-weight 1.0 \
  --dis-weight 0.2 \
  --div-weight 0.01

run_exp "20_lc_soft_crsid_full_m4_eta2" \
  "${SOFT_MAIN_ARGS[@]}"

# ---------------------------------------------------------------------------
# B. Soft SID ablations: prove local-consistent soft SID is useful and not just
# more candidates.
# ---------------------------------------------------------------------------

run_exp "21_soft_no_local_pruning_s000" \
  --model-variant crsid_soft \
  --cr-tail-tau 20 \
  --cr-residual-scale 1.0 \
  --cr-soft-top-m 4 \
  --cr-soft-min-overlap-slots 2 \
  --cr-soft-min-support 0.0 \
  --cr-soft-support-eta 2.0 \
  --cr-soft-hard-token-prior 1.0 \
  --cr-soft-reliability-floor 0.10 \
  --cr-soft-max-neighbors 50 \
  --sem-weight 1.0 \
  --dis-weight 0.2 \
  --div-weight 0.01

run_exp "22_soft_no_support_sharpen_eta1" \
  --model-variant crsid_soft \
  --cr-tail-tau 20 \
  --cr-residual-scale 1.0 \
  --cr-soft-top-m 4 \
  --cr-soft-min-overlap-slots 2 \
  --cr-soft-min-support 0.05 \
  --cr-soft-support-eta 1.0 \
  --cr-soft-hard-token-prior 1.0 \
  --cr-soft-reliability-floor 0.10 \
  --cr-soft-max-neighbors 50 \
  --sem-weight 1.0 \
  --dis-weight 0.2 \
  --div-weight 0.01

run_exp "23_soft_too_many_candidates_m8" \
  --model-variant crsid_soft \
  --cr-tail-tau 20 \
  --cr-residual-scale 1.0 \
  --cr-soft-top-m 8 \
  --cr-soft-min-overlap-slots 2 \
  --cr-soft-min-support 0.05 \
  --cr-soft-support-eta 2.0 \
  --cr-soft-hard-token-prior 1.0 \
  --cr-soft-reliability-floor 0.10 \
  --cr-soft-max-neighbors 50 \
  --sem-weight 1.0 \
  --dis-weight 0.2 \
  --div-weight 0.01

run_exp "24_soft_no_reliability_calib_rel1" \
  --model-variant crsid_soft \
  --cr-tail-tau 20 \
  --cr-residual-scale 1.0 \
  --cr-soft-top-m 4 \
  --cr-soft-min-overlap-slots 2 \
  --cr-soft-min-support 0.05 \
  --cr-soft-support-eta 2.0 \
  --cr-soft-hard-token-prior 1.0 \
  --cr-soft-reliability-floor 1.0 \
  --cr-soft-max-neighbors 50 \
  --sem-weight 1.0 \
  --dis-weight 0.2 \
  --div-weight 0.01

# ---------------------------------------------------------------------------
# C. Representation module ablations: prove basis/shared/private/adaptive
# residuals are necessary.
# ---------------------------------------------------------------------------

run_exp "30_soft_no_shared_residual" \
  "${SOFT_MAIN_ARGS[@]}" \
  --cr-disable-shared-residual

run_exp "31_soft_no_private_residual" \
  "${SOFT_MAIN_ARGS[@]}" \
  --cr-disable-private-residual

run_exp "32_soft_fixed_alpha_050" \
  "${SOFT_MAIN_ARGS[@]}" \
  --cr-alpha-override 0.5

if [[ "${RUN_HEAVY}" == "1" ]]; then
  run_exp "33_soft_no_semantic_basis" \
    "${SOFT_MAIN_ARGS[@]}" \
    --cr-disable-semantic-basis

  run_exp "34_soft_basis_only_no_residual" \
    "${SOFT_MAIN_ARGS[@]}" \
    --cr-residual-scale 0.0

  run_exp "35_soft_private_only_alpha_100" \
    "${SOFT_MAIN_ARGS[@]}" \
    --cr-alpha-override 1.0

  run_exp "36_soft_shared_only_alpha_000" \
    "${SOFT_MAIN_ARGS[@]}" \
    --cr-alpha-override 0.0
fi

"${PYTHON_BIN}" scripts/summarize_experiments.py \
  --root "${OUT_ROOT}" \
  --metric NDCG@10 \
  --top-k 50 \
  --csv "${OUT_ROOT}/summary.csv"

SCRIPT_END_TS="$(date +%s)"
SCRIPT_ELAPSED=$((SCRIPT_END_TS - SCRIPT_START_TS))
echo "============================================================"
echo "Finished LC-Soft CRSID ablations"
echo "Summary: ${OUT_ROOT}/summary.csv"
echo "Total elapsed: $(format_seconds "${SCRIPT_ELAPSED}")"
echo "============================================================"
