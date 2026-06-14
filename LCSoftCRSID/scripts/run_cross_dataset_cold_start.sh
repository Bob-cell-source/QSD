#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET_DIRS="${DATASET_DIRS:-runs/beauty runs/sports runs/toys_games}"
CORE_RUN_TAG="${CORE_RUN_TAG:-lcsoft_core_ablation_20260613}"
OUT_NAME="${OUT_NAME:-cold_start_comparison_20260614}"
SUMMARY_DIR="${SUMMARY_DIR:-runs/${OUT_NAME}_summary}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-128}"
CANDIDATE_CHUNK_SIZE="${CANDIDATE_CHUNK_SIZE:-512}"
COLD_THRESHOLD="${COLD_THRESHOLD:-5}"
BUCKETS="${BUCKETS:-0,1-2,3-5,6-10,>10}"
INCLUDE_ABLATIONS="${INCLUDE_ABLATIONS:-0}"
FORCE="${FORCE:-0}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

read -r -a DATASETS <<< "${DATASET_DIRS}"
RESULT_DIRS=()

first_existing() {
  local path
  for path in "$@"; do
    if [[ -f "${path}" ]]; then
      printf '%s\n' "${path}"
      return 0
    fi
  done
  return 1
}

for dataset_dir in "${DATASETS[@]}"; do
  echo "============================================================"
  echo "Cold-start evaluation: ${dataset_dir}"
  echo "============================================================"

  missing=0
  for required in sequences.json semantic_ids_rq.json stats.json; do
    if [[ ! -f "${dataset_dir}/${required}" ]]; then
      echo "Missing: ${dataset_dir}/${required}" >&2
      missing=1
    fi
  done
  if [[ "${missing}" == "1" ]]; then
    echo "Skip unavailable dataset: ${dataset_dir}" >&2
    continue
  fi

  core_root="${dataset_dir}/${CORE_RUN_TAG}"
  full_checkpoint="${core_root}/00_full_lcsoftcrsid/best.pt"
  if [[ ! -f "${full_checkpoint}" ]]; then
    echo "Missing final Full checkpoint: ${full_checkpoint}" >&2
    echo "Run LCSoftCRSID/scripts/run_core_ablations.sh first." >&2
    continue
  fi

  sasrec_checkpoint="$(first_existing \
    "${dataset_dir}/lc_soft_required_ablation/00_sasrec_id_only/best.pt" \
    "${dataset_dir}/crsid_soft_probe/00_sasrec_id_only/best.pt" \
    "${dataset_dir}/exp_sasrec/best.pt" \
    "${dataset_dir}/sasrec/best.pt" \
    || true)"
  if [[ -z "${sasrec_checkpoint}" ]]; then
    echo "No SASRec checkpoint found under ${dataset_dir}." >&2
    echo "Expected one of: lc_soft_required_ablation, crsid_soft_probe, exp_sasrec, sasrec." >&2
    continue
  fi

  output_dir="${dataset_dir}/${OUT_NAME}"
  output_json="${output_dir}/cold_start_metrics.json"
  RESULT_DIRS+=("${output_dir}")
  if [[ "${FORCE}" != "1" && -f "${output_json}" ]]; then
    echo "Skip existing: ${output_json}"
    continue
  fi

  checkpoints=(
    --checkpoint "full=${full_checkpoint}"
    --checkpoint "sasrec=${sasrec_checkpoint}"
  )

  if [[ "${INCLUDE_ABLATIONS}" == "1" ]]; then
    declare -a OPTIONAL_VARIANTS=(
      "hard_sid:${core_root}/10_hard_sid_m1/best.pt"
      "without_prior_bias:${core_root}/20_without_prior_bias/best.pt"
      "without_shared:${core_root}/30_without_shared_residual/best.pt"
      "without_private:${core_root}/31_without_private_residual/best.pt"
      "learnable_allocation:${core_root}/40_learnable_allocation/best.pt"
    )
    for spec in "${OPTIONAL_VARIANTS[@]}"; do
      label="${spec%%:*}"
      checkpoint="${spec#*:}"
      if [[ -f "${checkpoint}" ]]; then
        checkpoints+=(--checkpoint "${label}=${checkpoint}")
      else
        echo "Optional checkpoint missing: ${checkpoint}" >&2
      fi
    done
    unset OPTIONAL_VARIANTS
  fi

  echo "Full:   ${full_checkpoint}"
  echo "SASRec: ${sasrec_checkpoint}"
  echo "Output: ${output_dir}"

  "${PYTHON_BIN}" LCSoftCRSID/scripts/evaluate_cold_start.py \
    "${checkpoints[@]}" \
    --output-dir "${output_dir}" \
    --buckets "${BUCKETS}" \
    --cold-threshold "${COLD_THRESHOLD}" \
    --device "${DEVICE}" \
    --batch-size "${BATCH_SIZE}" \
    --candidate-chunk-size "${CANDIDATE_CHUNK_SIZE}"
done

if [[ "${#RESULT_DIRS[@]}" -eq 0 ]]; then
  echo "No dataset produced or exposed a cold-start result directory." >&2
  exit 1
fi

"${PYTHON_BIN}" LCSoftCRSID/scripts/summarize_cold_start.py \
  --result-dirs "${RESULT_DIRS[@]}" \
  --output-dir "${SUMMARY_DIR}" \
  --cold-group "cold_0-${COLD_THRESHOLD}" \
  --warm-group "warm_gt${COLD_THRESHOLD}"

echo "Combined CSV: ${SUMMARY_DIR}/summary.csv"
echo "LaTeX table: ${SUMMARY_DIR}/table.tex"
