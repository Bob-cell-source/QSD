#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <beauty|toys_games|sports> [batch_size]" >&2
  exit 2
fi

dataset="$1"
batch_size="${2:-512}"
dataset_dir="runs/${dataset}"
semantic_ids="${dataset_dir}/semantic_ids_rq.json"
output_dir="${dataset_dir}/locorec_loo_hard_centered_lift_20260614"

if [[ ! -f "${dataset_dir}/sequences.json" ]]; then
  echo "Missing ${dataset_dir}/sequences.json" >&2
  exit 1
fi
if [[ ! -f "${semantic_ids}" ]]; then
  echo "Missing ${semantic_ids}" >&2
  exit 1
fi

if [[ ! -f "${output_dir}/test_metrics.json" ]]; then
  python LoCoRecLOOCorrection/train.py \
    --dataset-dir "${dataset_dir}" \
    --semantic-ids "${semantic_ids}" \
    --output-dir "${output_dir}" \
    --device cuda \
    --epochs 100 \
    --early-stop-patience 10 \
    --batch-size "${batch_size}" \
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
    --soft-top-m 4 \
    --loo-min-overlap-slots 2 \
    --soft-min-support 0.05 \
    --soft-min-conditional-lift 0.0 \
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
else
  echo "Training result exists; skipping: ${output_dir}/test_metrics.json"
fi

if [[ ! -f "${output_dir}/grouped_evaluation/cold_start_metrics.json" ]]; then
  python LoCoRecLOOCorrection/evaluate_cold_start.py \
    --checkpoint "${output_dir}/best.pt" \
    --output-dir "${output_dir}/grouped_evaluation" \
    --device cuda \
    --batch-size "${batch_size}" \
    --candidate-chunk-size 256 \
    --cold-threshold 5 \
    --buckets "0,1-2,3-5,6-10,>10" \
    --popular-sid-quantile 0.90
else
  echo "Grouped result exists; skipping: ${output_dir}/grouped_evaluation/cold_start_metrics.json"
fi

echo "Result: ${output_dir}/test_metrics.json"
echo "Groups: ${output_dir}/grouped_evaluation/cold_start_metrics.json"
