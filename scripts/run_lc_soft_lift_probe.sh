#!/usr/bin/env bash
set -euo pipefail

# Small probe for the LC-SoftSID lift version.
# It does not overwrite previous best runs: outputs are written under
# ${DATASET_DIR}/lc_soft_lift_probe by default.

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASETS="${DATASETS:-runs/office runs/beauty runs/toys_games}"
DEVICE="${DEVICE:-cuda}"

EPOCHS="${EPOCHS:-30}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-5}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
EVAL_BATCH_EVAL_SIZE="${EVAL_BATCH_EVAL_SIZE:-1024}"
MAX_LEN="${MAX_LEN:-50}"
DIM="${DIM:-128}"
NUM_RANDOM_NEG="${NUM_RANDOM_NEG:-100}"
SEED="${SEED:-2026}"
FORCE="${FORCE:-0}"
RUN_HARD_BASELINE="${RUN_HARD_BASELINE:-1}"

SCRIPT_START_TS="$(date +%s)"

format_seconds() {
  local total="$1"
  local hours=$((total / 3600))
  local minutes=$(((total % 3600) / 60))
  local seconds=$((total % 60))
  printf "%02d:%02d:%02d" "${hours}" "${minutes}" "${seconds}"
}

run_exp() {
  local dataset_dir="$1"
  local name="$2"
  shift 2
  local semantic_ids="${SEMANTIC_IDS:-${dataset_dir}/semantic_ids_rq.json}"
  local out_root="${OUT_ROOT:-${dataset_dir}/lc_soft_lift_probe}"
  local output_dir="${out_root}/${name}"
  local start_ts
  local end_ts
  local elapsed

  if [[ ! -f "${dataset_dir}/sequences.json" ]]; then
    echo "Skip missing dataset: ${dataset_dir}"
    return
  fi
  if [[ ! -f "${semantic_ids}" ]]; then
    echo "Skip missing semantic IDs: ${semantic_ids}"
    return
  fi
  if [[ "${FORCE}" != "1" && -f "${output_dir}/test_metrics.json" ]]; then
    echo "Skip existing: ${output_dir}"
    return
  fi

  mkdir -p "${out_root}"
  start_ts="$(date +%s)"

  echo "============================================================"
  echo "Dataset: ${dataset_dir}"
  echo "Running: ${name}"
  echo "Output:  ${output_dir}"
  echo "Start:   $(date '+%Y-%m-%d %H:%M:%S')"
  echo "============================================================"

  "${PYTHON_BIN}" scripts/train_qsdrec.py \
    --dataset-dir "${dataset_dir}" \
    --semantic-ids "${semantic_ids}" \
    --output-dir "${output_dir}" \
    --device "${DEVICE}" \
    --epochs "${EPOCHS}" \
    --early-stop-patience "${EARLY_STOP_PATIENCE}" \
    --batch-size "${BATCH_SIZE}" \
    --eval-batch-eval-size "${EVAL_BATCH_EVAL_SIZE}" \
    --max-len "${MAX_LEN}" \
    --dim "${DIM}" \
    --num-hard-neg 0 \
    --num-random-neg "${NUM_RANDOM_NEG}" \
    --lr 0.001 \
    --weight-decay 0.0001 \
    --seed "${SEED}" \
    "$@"

  end_ts="$(date +%s)"
  elapsed=$((end_ts - start_ts))
  echo "Finished: ${name}"
  echo "Elapsed:  $(format_seconds "${elapsed}")"
  echo
}

run_dataset() {
  local dataset_dir="$1"
  local out_root="${OUT_ROOT:-${dataset_dir}/lc_soft_lift_probe}"

  echo "############################################################"
  echo "LC-SoftSID lift probe on ${dataset_dir}"
  echo "Output root: ${out_root}"
  echo "############################################################"

  if [[ "${RUN_HARD_BASELINE}" == "1" ]]; then
    run_exp "${dataset_dir}" "00_hard_crsid_tau20" \
      --model-variant crsid \
      --cr-tail-tau 20 \
      --cr-residual-scale 1.0 \
      --sem-weight 1.0 \
      --dis-weight 0.2 \
      --div-weight 0.01
  fi

  # Previous best-style soft setting: no lift, reliability not decoupled,
  # raw item frequency alpha. This preserves the current method for comparison.
  run_exp "${dataset_dir}" "10_soft_eta2_no_lift_rawfreq" \
    --model-variant crsid_soft \
    --cr-tail-tau 20 \
    --cr-residual-scale 1.0 \
    --cr-soft-top-m 4 \
    --cr-soft-min-overlap-slots 2 \
    --cr-soft-min-support 0.05 \
    --cr-soft-support-eta 2.0 \
    --cr-soft-hard-token-prior 1.0 \
    --cr-soft-reliability-floor 0.10 \
    --cr-soft-max-neighbors 50 \
    --cr-soft-lift-kappa 0.0 \
    --cr-alpha-frequency-transform raw \
    --sem-weight 1.0 \
    --dis-weight 0.2 \
    --div-weight 0.01

  # Local lift only: tests whether suppressing global SID popularity helps.
  run_exp "${dataset_dir}" "11_soft_eta2_lift_k1_rawfreq" \
    --model-variant crsid_soft \
    --cr-tail-tau 20 \
    --cr-residual-scale 1.0 \
    --cr-soft-top-m 4 \
    --cr-soft-min-overlap-slots 2 \
    --cr-soft-min-support 0.05 \
    --cr-soft-support-eta 2.0 \
    --cr-soft-hard-token-prior 1.0 \
    --cr-soft-reliability-floor 0.10 \
    --cr-soft-max-neighbors 50 \
    --cr-soft-lift-kappa 1.0 \
    --cr-soft-lift-clip 5.0 \
    --cr-alpha-frequency-transform raw \
    --sem-weight 1.0 \
    --dis-weight 0.2 \
    --div-weight 0.01

  # Lift + reliability decoupling: R_i no longer receives inflated support
  # from hard-token prior.
  run_exp "${dataset_dir}" "12_soft_eta2_lift_k1_decoupled_rawfreq" \
    --model-variant crsid_soft \
    --cr-tail-tau 20 \
    --cr-residual-scale 1.0 \
    --cr-soft-top-m 4 \
    --cr-soft-min-overlap-slots 2 \
    --cr-soft-min-support 0.05 \
    --cr-soft-support-eta 2.0 \
    --cr-soft-hard-token-prior 1.0 \
    --cr-soft-reliability-floor 0.10 \
    --cr-soft-max-neighbors 50 \
    --cr-soft-lift-kappa 1.0 \
    --cr-soft-lift-clip 5.0 \
    --cr-soft-decouple-reliability \
    --cr-alpha-frequency-transform raw \
    --sem-weight 1.0 \
    --dis-weight 0.2 \
    --div-weight 0.01

  # Full probe variant: lift + decoupled reliability + log-frequency alpha.
  run_exp "${dataset_dir}" "13_soft_eta2_lift_k1_decoupled_logfreq" \
    --model-variant crsid_soft \
    --cr-tail-tau 20 \
    --cr-residual-scale 1.0 \
    --cr-soft-top-m 4 \
    --cr-soft-min-overlap-slots 2 \
    --cr-soft-min-support 0.05 \
    --cr-soft-support-eta 2.0 \
    --cr-soft-hard-token-prior 1.0 \
    --cr-soft-reliability-floor 0.10 \
    --cr-soft-max-neighbors 50 \
    --cr-soft-lift-kappa 1.0 \
    --cr-soft-lift-clip 5.0 \
    --cr-soft-decouple-reliability \
    --cr-alpha-frequency-transform log \
    --sem-weight 1.0 \
    --dis-weight 0.2 \
    --div-weight 0.01

  "${PYTHON_BIN}" scripts/summarize_experiments.py \
    --root "${out_root}" \
    --metric NDCG@10 \
    --top-k 20 \
    --csv "${out_root}/summary.csv"
}

for dataset_dir in ${DATASETS}; do
  run_dataset "${dataset_dir}"
done

SCRIPT_END_TS="$(date +%s)"
SCRIPT_ELAPSED=$((SCRIPT_END_TS - SCRIPT_START_TS))
echo "============================================================"
echo "Finished LC-SoftSID lift probe"
echo "Datasets: ${DATASETS}"
echo "Total elapsed: $(format_seconds "${SCRIPT_ELAPSED}")"
echo "============================================================"
