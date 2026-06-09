#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET_DIR="${DATASET_DIR:-runs/beauty}"
SEMANTIC_IDS="${SEMANTIC_IDS:-${DATASET_DIR}/semantic_ids_rq.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATASET_DIR}/lcsoftcrsid_clean/main_seed2026}"
DEVICE="${DEVICE:-cuda}"

"${PYTHON_BIN}" LCSoftCRSID/train.py \
  --dataset-dir "${DATASET_DIR}" \
  --semantic-ids "${SEMANTIC_IDS}" \
  --output-dir "${OUTPUT_DIR}" \
  --device "${DEVICE}" \
  --epochs "${EPOCHS:-100}" \
  --early-stop-patience "${EARLY_STOP_PATIENCE:-10}" \
  --batch-size "${BATCH_SIZE:-1024}" \
  --eval-candidate-chunk-size "${EVAL_CANDIDATE_CHUNK_SIZE:-2048}" \
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
  --frequency-transform raw \
  --soft-top-m 4 \
  --soft-min-overlap-slots 2 \
  --soft-min-support 0.05 \
  --soft-support-eta 2.0 \
  --soft-hard-token-prior 1.0 \
  --soft-reliability-floor 0.10 \
  --soft-max-neighbors 50 \
  --seed "${SEED:-2026}"
