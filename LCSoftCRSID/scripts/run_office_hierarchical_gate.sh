#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET_DIR="${DATASET_DIR:-runs/office}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATASET_DIR}/hierarchical_residual_gate_20260614}"

if [[ "${FORCE:-0}" != "1" && -f "${OUTPUT_DIR}/test_metrics.json" ]]; then
  echo "Skip existing: ${OUTPUT_DIR}/test_metrics.json"
  exit 0
fi

"${PYTHON_BIN}" LCSoftCRSID/train.py \
  --dataset-dir "${DATASET_DIR}" \
  --semantic-ids "${DATASET_DIR}/semantic_ids_rq.json" \
  --output-dir "${OUTPUT_DIR}" \
  --device "${DEVICE:-cuda}" \
  --epochs "${EPOCHS:-100}" \
  --early-stop-patience "${EARLY_STOP_PATIENCE:-10}" \
  --batch-size "${BATCH_SIZE:-256}" \
  --eval-candidate-chunk-size "${EVAL_CHUNK_SIZE:-256}" \
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
  --residual-scale 1.0 \
  --soft-neighbor-source sid_overlap \
  --soft-top-m 4 \
  --soft-min-overlap-slots 3 \
  --soft-min-support 0.05 \
  --soft-reliability-floor 0.10 \
  --soft-max-neighbors 50 \
  --candidate-weight-mode prior_guided \
  --alpha-mode fixed \
  --fusion-mode hierarchical_residual_gate \
  --gate-warmup-epochs "${GATE_WARMUP_EPOCHS:-10}" \
  --gate-lr-scale "${GATE_LR_SCALE:-0.1}" \
  --gate-correction-scale "${GATE_CORRECTION_SCALE:-0.5}" \
  --gate-kl-weight "${GATE_KL_WEIGHT:-0.05}" \
  --gate-private-weight "${GATE_PRIVATE_WEIGHT:-0.1}" \
  --gate-private-margin "${GATE_PRIVATE_MARGIN:-0.05}" \
  --seed "${SEED:-2026}"
