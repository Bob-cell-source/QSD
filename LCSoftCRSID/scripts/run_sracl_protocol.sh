#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-100}"
SEEDS=(${SEEDS:-2026})
DATASETS=(${DATASETS:-office beauty sports})

for dataset in "${DATASETS[@]}"; do
  dataset_dir="runs/${dataset}"
  semantic_ids="${dataset_dir}/semantic_ids_rq.json"
  if [[ ! -f "${dataset_dir}/sequences.json" || ! -f "${semantic_ids}" ]]; then
    echo "Skip unavailable dataset: ${dataset_dir}"
    continue
  fi

  for seed in "${SEEDS[@]}"; do
    output_dir="${dataset_dir}/sracl_protocol/locorec_seed${seed}"
    if [[ "${FORCE:-0}" != "1" && -f "${output_dir}/test_metrics.json" ]]; then
      echo "Skip existing: ${output_dir}/test_metrics.json"
      continue
    fi

    "${PYTHON_BIN}" LCSoftCRSID/train.py \
      --dataset-dir "${dataset_dir}" \
      --semantic-ids "${semantic_ids}" \
      --output-dir "${output_dir}" \
      --device "${DEVICE}" \
      --epochs "${EPOCHS}" \
      --early-stop-patience 10 \
      --early-stop-metric MRR@10 \
      --batch-size 256 \
      --eval-candidate-chunk-size "${EVAL_CHUNK_SIZE:-512}" \
      --max-len 20 \
      --dim 128 \
      --num-heads 2 \
      --num-layers 2 \
      --dropout 0.5 \
      --lr 0.001 \
      --weight-decay 0.0 \
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
      --seed "${seed}"
  done
done
