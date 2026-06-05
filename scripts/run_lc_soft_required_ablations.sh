#!/usr/bin/env bash
set -euo pipefail

# Minimal paper ablations for LC-SoftSID / CRSID.
# Defaults run Beauty, Sports, and Toys/Games. This script intentionally keeps
# only necessary ablations for the thesis table.

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASETS="${DATASETS:-runs/beauty runs/sports runs/toys_games}"
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
  local out_root="${OUT_ROOT:-${dataset_dir}/lc_soft_required_ablation}"
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
  local out_root="${OUT_ROOT:-${dataset_dir}/lc_soft_required_ablation}"

  echo "############################################################"
  echo "Required LC-SoftSID ablations on ${dataset_dir}"
  echo "Output root: ${out_root}"
  echo "############################################################"

  # 1. Pure ID baseline. Shows whether the semantic-ID representation helps.
  run_exp "${dataset_dir}" "00_sasrec_id_only" \
    --model-variant qsdrec \
    --num-interests 1 \
    --sem-weight 0 \
    --dis-weight 0 \
    --div-weight 0

  # 2. Hard CRSID. Isolates the gain from soft SID over hard Semantic ID.
  run_exp "${dataset_dir}" "10_hard_crsid" \
    --model-variant crsid \
    --cr-tail-tau 20 \
    --cr-residual-scale 1.0 \
    --sem-weight 1.0 \
    --dis-weight 0.2 \
    --div-weight 0.01

  # 3. Full method. This is the current final method, without lift or behavior
  # neighbors by default.
  run_exp "${dataset_dir}" "20_lc_soft_full" \
    --model-variant crsid_soft \
    --cr-tail-tau 20 \
    --cr-residual-scale 1.0 \
    --cr-alpha-frequency-transform raw \
    --cr-soft-top-m 4 \
    --cr-soft-min-overlap-slots 2 \
    --cr-soft-min-support 0.05 \
    --cr-soft-support-eta 2.0 \
    --cr-soft-hard-token-prior 1.0 \
    --cr-soft-reliability-floor 0.10 \
    --cr-soft-max-neighbors 50 \
    --cr-soft-lift-kappa 0.0 \
    --sem-weight 1.0 \
    --dis-weight 0.2 \
    --div-weight 0.01

  # 4. Remove local consistency pruning. Tests whether local support filtering
  # is necessary for soft SID.
  run_exp "${dataset_dir}" "21_soft_no_local_pruning" \
    --model-variant crsid_soft \
    --cr-tail-tau 20 \
    --cr-residual-scale 1.0 \
    --cr-alpha-frequency-transform raw \
    --cr-soft-top-m 4 \
    --cr-soft-min-overlap-slots 2 \
    --cr-soft-min-support 0.0 \
    --cr-soft-support-eta 2.0 \
    --cr-soft-hard-token-prior 1.0 \
    --cr-soft-reliability-floor 0.10 \
    --cr-soft-max-neighbors 50 \
    --cr-soft-lift-kappa 0.0 \
    --sem-weight 1.0 \
    --dis-weight 0.2 \
    --div-weight 0.01

  # 5. Remove support sharpening. Tests whether high local support should be
  # emphasized instead of using flat candidate weights.
  run_exp "${dataset_dir}" "22_soft_eta1_no_sharpen" \
    --model-variant crsid_soft \
    --cr-tail-tau 20 \
    --cr-residual-scale 1.0 \
    --cr-alpha-frequency-transform raw \
    --cr-soft-top-m 4 \
    --cr-soft-min-overlap-slots 2 \
    --cr-soft-min-support 0.05 \
    --cr-soft-support-eta 1.0 \
    --cr-soft-hard-token-prior 1.0 \
    --cr-soft-reliability-floor 0.10 \
    --cr-soft-max-neighbors 50 \
    --cr-soft-lift-kappa 0.0 \
    --sem-weight 1.0 \
    --dis-weight 0.2 \
    --div-weight 0.01

  # 6. Remove shared semantic residual. Tests semantic transfer for long-tail
  # and shared-SID items.
  run_exp "${dataset_dir}" "30_no_shared_residual" \
    --model-variant crsid_soft \
    --cr-tail-tau 20 \
    --cr-residual-scale 1.0 \
    --cr-alpha-frequency-transform raw \
    --cr-soft-top-m 4 \
    --cr-soft-min-overlap-slots 2 \
    --cr-soft-min-support 0.05 \
    --cr-soft-support-eta 2.0 \
    --cr-soft-hard-token-prior 1.0 \
    --cr-soft-reliability-floor 0.10 \
    --cr-soft-max-neighbors 50 \
    --cr-soft-lift-kappa 0.0 \
    --cr-disable-shared-residual \
    --sem-weight 1.0 \
    --dis-weight 0.2 \
    --div-weight 0.01

  # 7. Remove private item residual. Tests whether item-specific memorization
  # is needed to avoid semantic drift.
  run_exp "${dataset_dir}" "31_no_private_residual" \
    --model-variant crsid_soft \
    --cr-tail-tau 20 \
    --cr-residual-scale 1.0 \
    --cr-alpha-frequency-transform raw \
    --cr-soft-top-m 4 \
    --cr-soft-min-overlap-slots 2 \
    --cr-soft-min-support 0.05 \
    --cr-soft-support-eta 2.0 \
    --cr-soft-hard-token-prior 1.0 \
    --cr-soft-reliability-floor 0.10 \
    --cr-soft-max-neighbors 50 \
    --cr-soft-lift-kappa 0.0 \
    --cr-disable-private-residual \
    --sem-weight 1.0 \
    --dis-weight 0.2 \
    --div-weight 0.01

  # 8. Add train-only behavior neighbors. This is the minimal ablation for the
  # user-side Semantic ID supplement: if it helps, behavior co-occurrence can
  # repair hard-SID under-sharing that SID-overlap neighbors miss.
  run_exp "${dataset_dir}" "40_with_behavior_neighbors_w050" \
    --model-variant crsid_soft \
    --cr-tail-tau 20 \
    --cr-residual-scale 1.0 \
    --cr-alpha-frequency-transform raw \
    --cr-soft-top-m 4 \
    --cr-soft-min-overlap-slots 2 \
    --cr-soft-min-support 0.05 \
    --cr-soft-support-eta 2.0 \
    --cr-soft-hard-token-prior 1.0 \
    --cr-soft-reliability-floor 0.10 \
    --cr-soft-max-neighbors 50 \
    --cr-soft-lift-kappa 0.0 \
    --cr-soft-behavior-weight 0.50 \
    --cr-soft-behavior-window 5 \
    --cr-soft-behavior-min-count 2 \
    --cr-soft-max-behavior-neighbors 50 \
    --sem-weight 1.0 \
    --dis-weight 0.2 \
    --div-weight 0.01

  "${PYTHON_BIN}" scripts/summarize_experiments.py \
    --root "${out_root}" \
    --metric NDCG@10 \
    --top-k 30 \
    --csv "${out_root}/summary.csv"
}

for dataset_dir in ${DATASETS}; do
  run_dataset "${dataset_dir}"
done

SCRIPT_END_TS="$(date +%s)"
SCRIPT_ELAPSED=$((SCRIPT_END_TS - SCRIPT_START_TS))
echo "============================================================"
echo "Finished required LC-SoftSID ablations"
echo "Datasets: ${DATASETS}"
echo "Total elapsed: $(format_seconds "${SCRIPT_ELAPSED}")"
echo "============================================================"
