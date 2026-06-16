#!/usr/bin/env bash
set -euo pipefail

# Reproduction entry for the paper/protocol requested by the user:
# "A Non-Contrastive Learning Framework for Sequential Recommendation with
# Preference-Preserving Profile Generation" on Amazon Office, Beauty, Sports.
#
# The local repository does not contain the original paper implementation.
# This script therefore runs the repository's protocol-compatible reproduction
# stack: Amazon 5-core preprocessing, Semantic ID construction from either
# local BGE vectors or a local encoder, a SASRec control, and the local non-contrastive LoCoRec variant under the same
# leave-one-out full-catalog evaluation protocol.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
DATASETS="${DATASETS:-office beauty sports}"
SEEDS="${SEEDS:-2021 2022 2023 2024 2025}"
RUN_TAG="${RUN_TAG:-pppg_reproduction}"
ENCODER_MODEL="${ENCODER_MODEL:-BAAI/bge-small-en-v1.5}"
CODEBOOK_SIZES="${CODEBOOK_SIZES:-64,128,256,512}"
SEMANTIC_BACKEND="${SEMANTIC_BACKEND:-encoder}"
FORCE="${FORCE:-0}"
SKIP_PREPROCESS="${SKIP_PREPROCESS:-0}"
SKIP_SEMANTIC_ID="${SKIP_SEMANTIC_ID:-0}"
RUN_SASREC="${RUN_SASREC:-1}"
RUN_LOCOREC="${RUN_LOCOREC:-1}"

dataset_reviews() {
  case "$1" in
    office) echo "data/reviews_Office_Products.json" ;;
    beauty) echo "data/reviews_beauty.json" ;;
    sports) echo "data/Sports_and_Outdoors.jsonl.gz" ;;
    *) return 1 ;;
  esac
}

dataset_meta() {
  case "$1" in
    office) echo "data/meta_Office_Products.json" ;;
    beauty) echo "data/meta_Beauty.json" ;;
    sports) echo "data/meta_Sports_and_Outdoors.jsonl.gz" ;;
    *) return 1 ;;
  esac
}

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "Missing required file: ${path}" >&2
    return 1
  fi
}

preprocess_dataset() {
  local dataset="$1"
  local dataset_dir="runs/${dataset}"
  local reviews
  local meta
  reviews="$(dataset_reviews "${dataset}")"
  meta="$(dataset_meta "${dataset}")"
  require_file "${reviews}"
  require_file "${meta}"

  if [[ "${SKIP_PREPROCESS}" == "1" && -f "${dataset_dir}/sequences.json" ]]; then
    echo "Use existing processed dataset: ${dataset_dir}"
    return
  fi
  if [[ "${FORCE}" != "1" && -f "${dataset_dir}/sequences.json" && -f "${dataset_dir}/stats.json" ]]; then
    echo "Processed dataset exists: ${dataset_dir}"
    return
  fi

  echo "Preprocess ${dataset}"
  "${PYTHON_BIN}" scripts/preprocess_amazon.py \
    --reviews "${reviews}" \
    --meta "${meta}" \
    --output-dir "${dataset_dir}" \
    --min-user-inter 5 \
    --min-item-inter 5
}

build_semantic_id() {
  local dataset="$1"
  local dataset_dir="runs/${dataset}"
  local semantic_ids="${dataset_dir}/semantic_ids_rq.json"

  if [[ "${SKIP_SEMANTIC_ID}" == "1" ]]; then
    require_file "${semantic_ids}"
    return
  fi
  if [[ "${FORCE}" != "1" && -f "${semantic_ids}" ]]; then
    echo "Semantic IDs exist: ${semantic_ids}"
    return
  fi

  require_file "${dataset_dir}/item_meta.json"
  if [[ "${SEMANTIC_BACKEND}" == "encoder" ]]; then
    echo "Build Semantic IDs for ${dataset} with ${ENCODER_MODEL}"
    "${PYTHON_BIN}" scripts/build_semantic_ids.py build \
      --item-meta "${dataset_dir}/item_meta.json" \
      --output "${semantic_ids}" \
      --encoder-model "${ENCODER_MODEL}" \
      --codebook-sizes "${CODEBOOK_SIZES}" \
      --batch-size "${SEM_BATCH_SIZE:-64}" \
      --max-length "${SEM_MAX_LENGTH:-512}" \
      --save-embeddings "${dataset_dir}/item_text_embeddings.npy" \
      --save-item-ids "${dataset_dir}/embedding_item_ids.json"
  elif [[ "${SEMANTIC_BACKEND}" == "tfidf" ]]; then
    echo "Build offline TF-IDF Semantic IDs for ${dataset}"
    "${PYTHON_BIN}" scripts/build_semantic_ids.py tfidf \
      --item-meta "${dataset_dir}/item_meta.json" \
      --output "${semantic_ids}" \
      --codebook-sizes "${CODEBOOK_SIZES}" \
      --max-features "${TFIDF_MAX_FEATURES:-50000}" \
      --svd-dim "${TFIDF_SVD_DIM:-256}" \
      --save-embeddings "${dataset_dir}/item_text_embeddings.npy" \
      --save-item-ids "${dataset_dir}/embedding_item_ids.json"
  elif [[ "${SEMANTIC_BACKEND}" == "rq" ]]; then
    local embeddings="${EMBEDDINGS_PATH:-${dataset_dir}/item_text_embeddings.npy}"
    local item_ids="${EMBEDDING_ITEM_IDS_PATH:-${dataset_dir}/embedding_item_ids.json}"
    require_file "${embeddings}"
    require_file "${item_ids}"
    "${PYTHON_BIN}" scripts/check_embedding_alignment.py \
      --item-meta "${dataset_dir}/item_meta.json" \
      --embeddings "${embeddings}" \
      --item-ids "${item_ids}"
    echo "Build Semantic IDs from existing embeddings for ${dataset}: ${embeddings}"
    "${PYTHON_BIN}" scripts/build_semantic_ids.py rq-kmeans \
      --embeddings "${embeddings}" \
      --item-ids "${item_ids}" \
      --output "${semantic_ids}" \
      --codebook-sizes "${CODEBOOK_SIZES}"
  else
    echo "Unsupported SEMANTIC_BACKEND=${SEMANTIC_BACKEND}; use rq, encoder, or tfidf." >&2
    return 1
  fi
}

run_sasrec() {
  local dataset="$1"
  local seed="$2"
  local dataset_dir="runs/${dataset}"
  local out="${dataset_dir}/${RUN_TAG}/sasrec/seed${seed}"
  local metrics="${out}/test_metrics.json"
  if [[ "${FORCE}" != "1" && -f "${metrics}" ]]; then
    echo "Skip existing SASRec: ${metrics}"
    return
  fi

  echo "Run SASRec: dataset=${dataset}, seed=${seed}"
  "${PYTHON_BIN}" scripts/train_qsdrec.py \
    --dataset-dir "${dataset_dir}" \
    --semantic-ids "${dataset_dir}/semantic_ids_rq.json" \
    --output-dir "${out}" \
    --device "${DEVICE}" \
    --model-variant qsdrec \
    --epochs "${EPOCHS:-100}" \
    --early-stop-patience "${EARLY_STOP_PATIENCE:-10}" \
    --early-stop-metric MRR@10 \
    --batch-size "${BATCH_SIZE:-256}" \
    --max-len "${MAX_LEN:-20}" \
    --dim "${DIM:-64}" \
    --num-heads "${NUM_HEADS:-2}" \
    --num-layers "${NUM_LAYERS:-2}" \
    --dropout "${DROPOUT:-0.5}" \
    --lr "${LR:-0.001}" \
    --weight-decay "${WEIGHT_DECAY:-0}" \
    --train-objective full_softmax \
    --train-candidate-chunk-size "${TRAIN_CHUNK_SIZE:-4096}" \
    --eval-batch-eval-size "${EVAL_CHUNK_SIZE:-1024}" \
    --keep-seen-items \
    --num-random-neg 0 \
    --num-hard-neg 0 \
    --num-interests 1 \
    --sem-weight 0 \
    --dis-weight 0 \
    --div-weight 0 \
    --seed "${seed}"
}

run_locorec() {
  local dataset="$1"
  local seed="$2"
  local dataset_dir="runs/${dataset}"
  local out="${dataset_dir}/${RUN_TAG}/locorec/seed${seed}"
  local metrics="${out}/test_metrics.json"
  if [[ "${FORCE}" != "1" && -f "${metrics}" ]]; then
    echo "Skip existing LoCoRec: ${metrics}"
    return
  fi

  echo "Run LoCoRec: dataset=${dataset}, seed=${seed}"
  "${PYTHON_BIN}" LCSoftCRSID/train.py \
    --dataset-dir "${dataset_dir}" \
    --semantic-ids "${dataset_dir}/semantic_ids_rq.json" \
    --output-dir "${out}" \
    --device "${DEVICE}" \
    --epochs "${EPOCHS:-100}" \
    --early-stop-patience "${EARLY_STOP_PATIENCE:-10}" \
    --early-stop-metric MRR@10 \
    --batch-size "${BATCH_SIZE:-256}" \
    --eval-candidate-chunk-size "${EVAL_CHUNK_SIZE:-1024}" \
    --max-len "${MAX_LEN:-20}" \
    --dim "${LOCOREC_DIM:-128}" \
    --num-heads "${NUM_HEADS:-2}" \
    --num-layers "${NUM_LAYERS:-2}" \
    --dropout "${DROPOUT:-0.5}" \
    --lr "${LR:-0.001}" \
    --weight-decay "${WEIGHT_DECAY:-0}" \
    --grad-clip "${GRAD_CLIP:-5.0}" \
    --num-random-negatives 0 \
    --train-objective full_softmax \
    --keep-seen-items \
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
}

for dataset in ${DATASETS}; do
  preprocess_dataset "${dataset}"
  build_semantic_id "${dataset}"
  require_file "runs/${dataset}/sequences.json"
  require_file "runs/${dataset}/semantic_ids_rq.json"

  for seed in ${SEEDS}; do
    if [[ "${RUN_SASREC}" == "1" ]]; then
      run_sasrec "${dataset}" "${seed}"
    fi
    if [[ "${RUN_LOCOREC}" == "1" ]]; then
      run_locorec "${dataset}" "${seed}"
    fi
  done
done

"${PYTHON_BIN}" scripts/summarize_pppg_reproduction.py \
  --run-tag "${RUN_TAG}" \
  --datasets ${DATASETS} \
  --output-dir "runs/${RUN_TAG}_summary"
DATASETS="office" \
  SEEDS="2025" \
  RUN_TAG=nclsr_mce_local_protocol \
  DROPOUT=0.2 \
  bash scripts/run_nclsr_local_protocol.sh