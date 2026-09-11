#!/usr/bin/env bash
set -euo pipefail

# One-shot reproducible run for the current LoCoRec/Hard-SID comparison.
# Run from the repository root, or set PROJECT_ROOT explicitly.
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
SEEDS="${SEEDS:-2026 2027 2028 2029 2030}"
DATASETS="${DATASETS:-beauty sports toys_games}"
RUN_TAG="${RUN_TAG:-sracl_aligned_20260911}"
ENCODER_MODEL="${ENCODER_MODEL:-BAAI/bge-small-en-v1.5}"
THREADS="${THREADS:-2}"
BUILD_SEMANTIC_IDS="${BUILD_SEMANTIC_IDS:-1}"
RUN_GROUPED="${RUN_GROUPED:-1}"

die() { echo "ERROR: $*" >&2; exit 1; }

for dataset in ${DATASETS}; do
  dataset_dir="runs/${dataset}"
  [[ -d "${dataset_dir}" ]] || die "Missing dataset directory: ${dataset_dir}. Prepare the raw processed dataset first."
  for file in sequences.json stats.json item_meta.json; do
    [[ -f "${dataset_dir}/${file}" ]] || die "Missing ${dataset_dir}/${file}"
  done

  semantic_ids="${dataset_dir}/semantic_ids_rq.json"
  if [[ "${BUILD_SEMANTIC_IDS}" == "1" && ! -f "${semantic_ids}" ]]; then
    echo "[${dataset}] Building Semantic IDs"
    "${PYTHON_BIN}" scripts/build_semantic_ids.py build \
      --item-meta "${dataset_dir}/item_meta.json" \
      --output "${semantic_ids}" \
      --encoder-model "${ENCODER_MODEL}" \
      --codebook-sizes 64,128,256,512 \
      --batch-size 64 \
      --max-length 512 \
      --save-embeddings "${dataset_dir}/item_text_embeddings.npy" \
      --save-item-ids "${dataset_dir}/embedding_item_ids.json"
  fi
  [[ -f "${semantic_ids}" ]] || die "Missing ${semantic_ids}; set BUILD_SEMANTIC_IDS=1 or provide it."

  output_dir="${dataset_dir}/${RUN_TAG}"
  echo "[${dataset}] Training Hard-SID and LoCoRec"
  OMP_NUM_THREADS="${THREADS}" "${PYTHON_BIN}" -m LoCoRecSimple.experiment \
    --dataset-dir "${dataset_dir}" \
    --semantic-ids "${semantic_ids}" \
    --output-dir "${output_dir}" \
    --protocol sracl \
    --variants hard full \
    --seeds ${SEEDS} \
    --device "${DEVICE}" \
    --epochs 100 \
    --patience 10 \
    --threads "${THREADS}"

  if [[ "${RUN_GROUPED}" == "1" ]]; then
    for seed in ${SEEDS}; do
      hard_ckpt="${output_dir}/hard/seed${seed}/best.pt"
      full_ckpt="${output_dir}/full/seed${seed}/best.pt"
      [[ -f "${hard_ckpt}" && -f "${full_ckpt}" ]] || die "Incomplete checkpoints for ${dataset}, seed ${seed}"
      "${PYTHON_BIN}" -m LoCoRecSimple.evaluate_grouped \
        --dataset-dir "${dataset_dir}" \
        --semantic-ids "${semantic_ids}" \
        --checkpoint "hard=${hard_ckpt}" \
        --checkpoint "locorec=${full_ckpt}" \
        --output "${output_dir}/grouped_seed${seed}.json" \
        --device "${DEVICE}" \
        --batch-size 256
    done
  fi
done

echo "All requested datasets completed. Main results are in runs/{dataset}/${RUN_TAG}/summary.json."
echo "Grouped results are in runs/{dataset}/${RUN_TAG}/grouped_seed*.json."
