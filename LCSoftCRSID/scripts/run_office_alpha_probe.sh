#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET_DIR="${DATASET_DIR:-runs/office}"
SEMANTIC_IDS="${SEMANTIC_IDS:-${DATASET_DIR}/semantic_ids_rq.json}"
OUT_ROOT="${OUT_ROOT:-${DATASET_DIR}/lcsoftcrsid_alpha_probe_v1}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-100}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-10}"
BATCH_SIZE="${BATCH_SIZE:-512}"
EVAL_CANDIDATE_CHUNK_SIZE="${EVAL_CANDIDATE_CHUNK_SIZE:-1024}"
SEED="${SEED:-2026}"
FORCE="${FORCE:-0}"

run_exp() {
  local name="$1"
  shift
  local output_dir="${OUT_ROOT}/${name}"
  if [[ "${FORCE}" != "1" && -f "${output_dir}/test_metrics.json" ]]; then
    echo "Skip existing: ${output_dir}"
    return
  fi

  echo "============================================================"
  echo "Running: ${name}"
  echo "Output:  ${output_dir}"
  echo "============================================================"
  "${PYTHON_BIN}" LCSoftCRSID/train.py \
    --dataset-dir "${DATASET_DIR}" \
    --semantic-ids "${SEMANTIC_IDS}" \
    --output-dir "${output_dir}" \
    --device "${DEVICE}" \
    --epochs "${EPOCHS}" \
    --early-stop-patience "${EARLY_STOP_PATIENCE}" \
    --batch-size "${BATCH_SIZE}" \
    --eval-candidate-chunk-size "${EVAL_CANDIDATE_CHUNK_SIZE}" \
    --max-len 50 \
    --dim 128 \
    --num-heads 2 \
    --num-layers 2 \
    --dropout 0.2 \
    --lr 0.001 \
    --weight-decay 0.0001 \
    --grad-clip 5.0 \
    --num-random-negatives 100 \
    --tail-tau 20 \
    --soft-top-m 4 \
    --soft-min-overlap-slots 2 \
    --soft-min-support 0.05 \
    --soft-reliability-floor 0.10 \
    --soft-max-neighbors 50 \
    --seed "${SEED}" \
    "$@"
}

# Current main mechanism: fixed local candidate weights and fixed alpha formula.
run_exp "00_fixed_alpha" \
  --candidate-weight-mode fixed \
  --alpha-mode fixed

# Isolated test: only alpha becomes learnable; everything else stays fixed.
run_exp "10_learnable_monotonic_alpha" \
  --candidate-weight-mode fixed \
  --alpha-mode learnable_monotonic

# Diagnostic interaction test. Do not select this as the main variant unless it
# beats the isolated alpha version on validation NDCG@10.
run_exp "20_learnable_alpha_prior_attention" \
  --candidate-weight-mode prior_guided \
  --alpha-mode learnable_monotonic

"${PYTHON_BIN}" LCSoftCRSID/summarize_alpha_probe.py --root "${OUT_ROOT}"
