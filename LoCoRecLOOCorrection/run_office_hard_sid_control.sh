#!/usr/bin/env bash
set -euo pipefail

# Hard SID baseline: direct fusion of deterministic SID semantics and Item-ID
# collaborative embeddings. No Soft SID candidates, LOO statistics, residual
# decomposition, or adaptive gates are used.
dropout="${1:-0.2}"
dropout_tag="${dropout//./p}"

python LoCoRecLOOCorrection/train.py \
  --dataset-dir runs/office \
  --semantic-ids runs/office/semantic_ids_rq.json \
  --output-dir "runs/office/hard_sid_direct_fusion_d${dropout_tag}_20260614" \
  --model-variant hard_sid_fusion \
  --device cuda \
  --epochs 100 \
  --early-stop-patience 10 \
  --batch-size 256 \
  --eval-candidate-chunk-size 256 \
  --num-workers 0 \
  --max-len 50 \
  --dim 64 \
  --num-heads 2 \
  --num-layers 2 \
  --dropout "${dropout}" \
  --lr 0.001 \
  --weight-decay 0.0001 \
  --grad-clip 5.0 \
  --num-random-negatives 100 \
  --seed 2026
