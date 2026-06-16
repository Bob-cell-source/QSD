#!/usr/bin/env bash
set -euo pipefail

# Paper implementation of LoCoRec:
#   1. SID-overlap local candidates and statistical priors;
#   2. learnable prior-guided attention;
#   3. semantic basis plus shared/private collaborative residuals;
#   4. hierarchical residual gate with bounded corrections.
#
# This entry point intentionally excludes LOO candidate construction, lift,
# hard-centered correction, behavior neighbors, and flat three-way gates.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
DATASETS="${DATASETS:-office beauty sports}"
SEEDS="${SEEDS:-2026}"
RUN_TAG="${RUN_TAG:-locorec_paper}"
FORCE="${FORCE:-0}"

for dataset in ${DATASETS}; do
  dataset_dir="runs/${dataset}"
  semantic_ids="${dataset_dir}/semantic_ids_rq.json"

  if [[ ! -f "${dataset_dir}/sequences.json" || ! -f "${semantic_ids}" ]]; then
    echo "Skip unavailable dataset: ${dataset_dir}" >&2
    continue
  fi

  for seed in ${SEEDS}; do
    output_dir="${dataset_dir}/${RUN_TAG}/seed${seed}"
    metrics="${output_dir}/test_metrics.json"

    if [[ "${FORCE}" != "1" && -f "${metrics}" ]]; then
      echo "Skip existing: ${metrics}"
      continue
    fi

    echo "Running LoCoRec: dataset=${dataset}, seed=${seed}"
    "${PYTHON_BIN}" LCSoftCRSID/train.py \
      --dataset-dir "${dataset_dir}" \
      --semantic-ids "${semantic_ids}" \
      --output-dir "${output_dir}" \
      --device "${DEVICE}" \
      --epochs "${EPOCHS:-100}" \
      --early-stop-patience "${EARLY_STOP_PATIENCE:-10}" \
      --early-stop-metric NDCG@10 \
      --batch-size "${BATCH_SIZE:-256}" \
      --eval-candidate-chunk-size "${EVAL_CHUNK_SIZE:-512}" \
      --num-workers "${NUM_WORKERS:-0}" \
      --max-len "${MAX_LEN:-50}" \
      --dim "${DIM:-128}" \
      --num-heads "${NUM_HEADS:-2}" \
      --num-layers "${NUM_LAYERS:-2}" \
      --dropout "${DROPOUT:-0.2}" \
      --lr "${LR:-0.001}" \
      --weight-decay "${WEIGHT_DECAY:-0.0001}" \
      --grad-clip "${GRAD_CLIP:-5.0}" \
      --num-random-negatives "${NUM_RANDOM_NEGATIVES:-100}" \
      --train-objective sampled \
      --tail-tau "${TAIL_TAU:-20}" \
      --residual-scale "${RESIDUAL_SCALE:-1.0}" \
      --soft-neighbor-source sid_overlap \
      --soft-top-m "${SOFT_TOP_M:-4}" \
      --soft-min-overlap-slots "${SOFT_MIN_OVERLAP_SLOTS:-3}" \
      --soft-min-support "${SOFT_MIN_SUPPORT:-0.05}" \
      --soft-reliability-floor "${SOFT_RELIABILITY_FLOOR:-0.10}" \
      --soft-max-neighbors "${SOFT_MAX_NEIGHBORS:-50}" \
      --candidate-weight-mode prior_guided \
      --alpha-mode fixed \
      --fusion-mode hierarchical_residual_gate \
      --gate-warmup-epochs "${GATE_WARMUP_EPOCHS:-10}" \
      --gate-lr-scale "${GATE_LR_SCALE:-0.1}" \
      --gate-correction-scale "${GATE_CORRECTION_SCALE:-0.3}" \
      --gate-kl-weight "${GATE_KL_WEIGHT:-0.05}" \
      --gate-private-weight "${GATE_PRIVATE_WEIGHT:-0.1}" \
      --gate-private-margin "${GATE_PRIVATE_MARGIN:-0.05}" \
      --seed "${seed}"
  done
done
