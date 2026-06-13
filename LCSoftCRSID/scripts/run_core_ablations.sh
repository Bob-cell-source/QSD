#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET_DIRS="${DATASET_DIRS:-runs/beauty runs/sports runs/toys_games}"
RUN_TAG="${RUN_TAG:-lcsoft_core_ablation_20260613}"
SUMMARY_DIR="${SUMMARY_DIR:-runs/${RUN_TAG}_summary}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-100}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-10}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
EVAL_CHUNK_SIZE="${EVAL_CHUNK_SIZE:-1024}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SEED="${SEED:-2026}"
FORCE="${FORCE:-0}"

# Helps avoid allocator fragmentation. It does not reduce the actual model memory requirement.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

read -r -a DATASETS <<< "${DATASET_DIRS}"
SUMMARY_ROOTS=()

require_dataset() {
  local dataset_dir="$1"
  local missing=0
  for name in sequences.json semantic_ids_rq.json stats.json; do
    if [[ ! -f "${dataset_dir}/${name}" ]]; then
      echo "Missing required file: ${dataset_dir}/${name}" >&2
      missing=1
    fi
  done
  if [[ "${missing}" == "1" ]]; then
    return 1
  fi
}

run_variant() {
  local dataset_dir="$1"
  local output_root="$2"
  local name="$3"
  local top_m="$4"
  local candidate_mode="$5"
  local alpha_mode="$6"
  shift 6

  local output_dir="${output_root}/${name}"
  local metrics="${output_dir}/test_metrics.json"

  if [[ "${FORCE}" != "1" && -f "${metrics}" ]]; then
    echo "Skip existing: ${metrics}"
    return
  fi

  echo "============================================================"
  echo "Dataset: ${dataset_dir}"
  echo "Variant: ${name}"
  echo "Output:  ${output_dir}"
  echo "============================================================"

  "${PYTHON_BIN}" LCSoftCRSID/train.py \
    --dataset-dir "${dataset_dir}" \
    --semantic-ids "${dataset_dir}/semantic_ids_rq.json" \
    --output-dir "${output_dir}" \
    --device "${DEVICE}" \
    --epochs "${EPOCHS}" \
    --early-stop-patience "${EARLY_STOP_PATIENCE}" \
    --batch-size "${BATCH_SIZE}" \
    --eval-candidate-chunk-size "${EVAL_CHUNK_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --max-len 50 \
    --dim 128 \
    --num-heads 2 \
    --num-layers 2 \
    --dropout 0.2 \
    --lr 0.001 \
    --weight-decay 0.0001 \
    --grad-clip 5.0 \
    --num-random-negatives 100 \
    --tail-tau 20 \
    --soft-neighbor-source sid_overlap \
    --soft-top-m "${top_m}" \
    --soft-min-overlap-slots 3 \
    --soft-min-support 0.05 \
    --soft-reliability-floor 0.10 \
    --soft-max-neighbors 50 \
    --candidate-weight-mode "${candidate_mode}" \
    --alpha-mode "${alpha_mode}" \
    --seed "${SEED}" \
    "$@"
}

for dataset_dir in "${DATASETS[@]}"; do
  if ! require_dataset "${dataset_dir}"; then
    echo "Skip unavailable dataset: ${dataset_dir}" >&2
    continue
  fi

  output_root="${dataset_dir}/${RUN_TAG}"
  mkdir -p "${output_root}"
  SUMMARY_ROOTS+=("${output_root}")

  # Final method: local candidate prior, recommendation-guided attention,
  # and fixed frequency-reliability residual allocation.
  run_variant "${dataset_dir}" "${output_root}" \
    "00_full_lcsoftcrsid" 4 prior_guided fixed

  # M=1 collapses every slot to its anchored hard token.
  run_variant "${dataset_dir}" "${output_root}" \
    "10_hard_sid_m1" 1 prior_guided fixed

  # The candidate set is unchanged; only the explicit log-prior bias is removed.
  run_variant "${dataset_dir}" "${output_root}" \
    "20_without_prior_bias" 4 learned fixed

  run_variant "${dataset_dir}" "${output_root}" \
    "30_without_shared_residual" 4 prior_guided fixed \
    --disable-shared-residual

  run_variant "${dataset_dir}" "${output_root}" \
    "31_without_private_residual" 4 prior_guided fixed \
    --disable-private-residual

  run_variant "${dataset_dir}" "${output_root}" \
    "40_learnable_allocation" 4 prior_guided learnable_monotonic

  "${PYTHON_BIN}" LCSoftCRSID/scripts/summarize_core_ablations.py \
    --roots "${output_root}" \
    --output-dir "${output_root}"
done

if [[ "${#SUMMARY_ROOTS[@]}" -eq 0 ]]; then
  echo "No valid dataset directory was found." >&2
  exit 1
fi

"${PYTHON_BIN}" LCSoftCRSID/scripts/summarize_core_ablations.py \
  --roots "${SUMMARY_ROOTS[@]}" \
  --output-dir "${SUMMARY_DIR}"

echo "Core ablations complete. Combined summary: ${SUMMARY_DIR}/summary.csv"
