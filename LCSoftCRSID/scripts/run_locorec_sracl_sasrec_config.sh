#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET_DIR="${DATASET_DIR:-runs/office}"
SEED="${SEED:-2021}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATASET_DIR}/sracl_locorec_compatible/seed${SEED}}"

"${PYTHON_BIN}" LCSoftCRSID/train.py \
  --dataset-dir "${DATASET_DIR}" \
  --semantic-ids "${DATASET_DIR}/semantic_ids_rq.json" \
  --output-dir "${OUTPUT_DIR}" \
  --device "${DEVICE:-cuda}" \
  --epochs 100 \
  --early-stop-patience 10 \
  --early-stop-metric MRR@10 \
  --batch-size 256 \
  --eval-candidate-chunk-size "${EVAL_CHUNK_SIZE:-1024}" \
  --max-len 20 \
  --dim 128 \
  --num-heads 2 \
  --num-layers 2 \
  --dropout 0.5 \
  --lr 0.001 \
  --weight-decay 0 \
  --grad-clip 5.0 \
  --num-random-negatives 0 \
  --train-objective full_softmax \
  --keep-seen-items \
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
  --gate-warmup-epochs 10 \
  --gate-lr-scale 0.1 \
  --gate-correction-scale 0.3 \
  --gate-kl-weight 0.05 \
  --gate-private-weight 0.1 \
  --gate-private-margin 0.05 \
  --seed "${SEED}"
