#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
DATASETS="${DATASETS:-office beauty sports}"
SEEDS="${SEEDS:-2021 2022 2023 2024 2025}"
RUN_TAG="${RUN_TAG:-nclsr_local_protocol}"
FORCE="${FORCE:-0}"

for dataset in ${DATASETS}; do
  dataset_dir="runs/${dataset}"
  embeddings="${EMBEDDINGS_PATH:-${dataset_dir}/item_text_embeddings.npy}"
  item_ids="${EMBEDDING_ITEM_IDS_PATH:-${dataset_dir}/embedding_item_ids.json}"
  if [[ ! -f "${dataset_dir}/sequences.json" || ! -f "${dataset_dir}/stats.json" ]]; then
    echo "Skip ${dataset}: missing processed dataset under ${dataset_dir}" >&2
    continue
  fi
  if [[ ! -f "${embeddings}" || ! -f "${item_ids}" ]]; then
    echo "Skip ${dataset}: missing BGE embeddings ${embeddings} or ${item_ids}" >&2
    continue
  fi

  "${PYTHON_BIN}" scripts/check_embedding_alignment.py \
    --item-meta "${dataset_dir}/item_meta.json" \
    --embeddings "${embeddings}" \
    --item-ids "${item_ids}"

  for seed in ${SEEDS}; do
    output_dir="${dataset_dir}/${RUN_TAG}/seed${seed}"
    if [[ "${FORCE}" != "1" && -f "${output_dir}/test_metrics.json" ]]; then
      echo "Skip existing: ${output_dir}/test_metrics.json"
      continue
    fi
    echo "Run NCL-SR local protocol: dataset=${dataset}, seed=${seed}"
    "${PYTHON_BIN}" scripts/train_nclsr.py \
      --dataset-dir "${dataset_dir}" \
      --item-embeddings "${embeddings}" \
      --embedding-item-ids "${item_ids}" \
      --output-dir "${output_dir}" \
      --device "${DEVICE}" \
      --epochs "${EPOCHS:-100}" \
      --early-stop-patience "${EARLY_STOP_PATIENCE:-10}" \
      --early-stop-metric "${EARLY_STOP_METRIC:-NDCG@10}" \
      --batch-size "${BATCH_SIZE:-256}" \
      --max-len "${MAX_LEN:-50}" \
      --dim "${DIM:-0}" \
      --num-heads "${NUM_HEADS:-2}" \
      --num-layers "${NUM_LAYERS:-2}" \
      --dropout "${DROPOUT:-0.4}" \
      --lr "${LR:-0.001}" \
      --weight-decay "${WEIGHT_DECAY:-0.0001}" \
      --train-objective "${TRAIN_OBJECTIVE:-sampled}" \
      --num-random-negatives "${NUM_RANDOM_NEGATIVES:-100}" \
      --train-candidate-chunk-size "${TRAIN_CHUNK_SIZE:-4096}" \
      --eval-batch-eval-size "${EVAL_CHUNK_SIZE:-1024}" \
      --synonym-top-k "${SYNONYM_TOP_K:-20}" \
      --synonym-chunk-size "${SYNONYM_CHUNK_SIZE:-512}" \
      --replace-count "${REPLACE_COUNT:-3}" \
      --dp-epsilon "${DP_EPSILON:-1.0}" \
      --uniform-weight "${UNIFORM_WEIGHT:-0.05}" \
      --align-weight "${ALIGN_WEIGHT:-0.1}" \
      --mce-order "${MCE_ORDER:-4}" \
      --mce-mu "${MCE_MU:-1.0}" \
      --mce-lambda "${MCE_LAMBDA:-1.0}" \
      --dp-output "${DP_OUTPUT:-nearest}" \
      --score-temperature "${SCORE_TEMPERATURE:-0.07}" \
      --seed "${seed}"
  done
done
