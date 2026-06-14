#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET_DIR="${DATASET_DIR:-runs/office}"
SEED="${SEED:-2021}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATASET_DIR}/sracl_sasrec_compatible/seed${SEED}}"

"${PYTHON_BIN}" scripts/train_qsdrec.py \
  --dataset-dir "${DATASET_DIR}" \
  --semantic-ids "${DATASET_DIR}/semantic_ids_rq.json" \
  --output-dir "${OUTPUT_DIR}" \
  --device "${DEVICE:-cuda}" \
  --epochs 100 \
  --early-stop-patience 10 \
  --early-stop-metric MRR@10 \
  --batch-size 256 \
  --max-len 20 \
  --dim 64 \
  --num-heads 2 \
  --num-layers 2 \
  --dropout 0.5 \
  --lr 0.001 \
  --weight-decay 0 \
  --train-objective full_softmax \
  --train-candidate-chunk-size 4096 \
  --eval-batch-eval-size 1024 \
  --keep-seen-items \
  --num-random-neg 0 \
  --num-hard-neg 0 \
  --num-interests 1 \
  --sem-weight 0 \
  --dis-weight 0 \
  --div-weight 0 \
  --seed "${SEED}"
