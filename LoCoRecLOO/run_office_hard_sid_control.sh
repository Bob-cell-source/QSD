#!/usr/bin/env bash
set -euo pipefail

# Strict control for locorec_loo_delta2_20260614: only Soft SID top-M changes.
python LoCoRecLOO/train.py \
  --dataset-dir runs/office \
  --semantic-ids runs/office/semantic_ids_rq.json \
  --output-dir runs/office/locorec_loo_delta2_hard_m1_20260614 \
  --device cuda \
  --epochs 100 \
  --early-stop-patience 10 \
  --batch-size 256 \
  --eval-candidate-chunk-size 256 \
  --num-workers 0 \
  --max-len 50 \
  --dim 128 \
  --num-heads 2 \
  --num-layers 2 \
  --dropout 0.2 \
  --lr 0.001 \
  --weight-decay 0.0001 \
  --grad-clip 5.0 \
  --num-random-negatives 100 \
  --soft-top-m 1 \
  --loo-min-overlap-slots 2 \
  --soft-min-support 0.05 \
  --soft-max-neighbors 50 \
  --tail-tau 20.0 \
  --residual-scale 1.0 \
  --gate-correction-scale 0.3 \
  --gate-kl-weight 0.05 \
  --gate-private-weight 0.1 \
  --gate-private-margin 0.05 \
  --gate-warmup-epochs 10 \
  --gate-lr-scale 0.1 \
  --seed 2026
