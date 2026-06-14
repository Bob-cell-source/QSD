#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <beauty|toys_games|sports> [batch_size]" >&2
  exit 2
fi

dataset="$1"
batch_size="${2:-256}"
dataset_dir="runs/${dataset}"
semantic_ids="${dataset_dir}/semantic_ids_rq.json"
root="${dataset_dir}/locorec_loo_delta2_comparison_20260614"

if [[ ! -f "${dataset_dir}/sequences.json" || ! -f "${semantic_ids}" ]]; then
  echo "Missing processed dataset or Semantic IDs under ${dataset_dir}." >&2
  exit 1
fi

run_variant() {
  local name="$1"
  local top_m="$2"
  local output_dir="${root}/${name}"

  if [[ ! -f "${output_dir}/test_metrics.json" ]]; then
    python LoCoRecLOO/train.py \
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
      --soft-top-m "${top_m}" \
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
  fi

  if [[ ! -f "${output_dir}/grouped_evaluation/cold_start_metrics.json" ]]; then
    python LoCoRecLOO/evaluate_cold_start.py \
      --checkpoint "${output_dir}/best.pt" \
      --output-dir "${output_dir}/grouped_evaluation" \
      --device cuda \
      --batch-size "${batch_size}" \
      --candidate-chunk-size 256 \
      --cold-threshold 5 \
      --buckets "0,1-2,3-5,6-10,>10" \
      --popular-sid-quantile 0.90
  fi
}

run_variant "soft_m4" 4
run_variant "hard_m1" 1

python LoCoRecLOO/summarize_soft_hard.py \
  --soft "${root}/soft_m4/grouped_evaluation/cold_start_metrics.json" \
  --hard "${root}/hard_m1/grouped_evaluation/cold_start_metrics.json" \
  --output "${root}/soft_vs_hard.csv"

