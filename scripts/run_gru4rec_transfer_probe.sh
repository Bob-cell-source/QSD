#!/usr/bin/env bash
set -euo pipefail

# Transferability check: keep the sequential backbone fixed as GRU4Rec and
# compare ID-only item embeddings against the full LC-SoftCRSID representation.

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASETS="${DATASETS:-runs/office}"
DEVICE="${DEVICE:-cuda}"

EPOCHS="${EPOCHS:-100}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-10}"
BATCH_SIZE="${BATCH_SIZE:-256}"
EVAL_BATCH_EVAL_SIZE="${EVAL_BATCH_EVAL_SIZE:-256}"
MAX_LEN="${MAX_LEN:-50}"
DIM="${DIM:-128}"
NUM_LAYERS="${NUM_LAYERS:-2}"
NUM_RANDOM_NEG="${NUM_RANDOM_NEG:-100}"
SEED="${SEED:-2026}"
FORCE="${FORCE:-0}"

run_exp() {
  local dataset_dir="$1"
  local name="$2"
  shift 2

  local semantic_ids="${SEMANTIC_IDS:-${dataset_dir}/semantic_ids_rq.json}"
  local out_root="${OUT_ROOT:-${dataset_dir}/gru4rec_transfer_probe}"
  local output_dir="${out_root}/${name}"

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
  echo "============================================================"
  echo "Dataset: ${dataset_dir}"
  echo "Running: ${name}"
  echo "Output:  ${output_dir}"
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
    --num-layers "${NUM_LAYERS}" \
    --dropout 0.2 \
    --num-hard-neg 0 \
    --num-random-neg "${NUM_RANDOM_NEG}" \
    --lr 0.001 \
    --weight-decay 0.0001 \
    --seed "${SEED}" \
    "$@"
}

for dataset_dir in ${DATASETS}; do
  # Pure ID baseline under exactly the same GRU4Rec sequence encoder.
  run_exp "${dataset_dir}" "00_gru4rec_id_only" \
    --model-variant gru4rec \
    --sem-weight 0 \
    --dis-weight 0 \
    --div-weight 0

  # Full paper method: only the sequence encoder changes from SASRec to GRU4Rec.
  run_exp "${dataset_dir}" "10_gru4rec_lc_soft_full" \
    --model-variant gru4rec_lcsoft \
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
    --cr-soft-behavior-weight 0.0 \
    --sem-weight 1.0 \
    --dis-weight 0 \
    --div-weight 0

  out_root="${OUT_ROOT:-${dataset_dir}/gru4rec_transfer_probe}"
  "${PYTHON_BIN}" scripts/summarize_experiments.py \
    --root "${out_root}" \
    --metric NDCG@10 \
    --csv "${out_root}/summary.csv"
done
