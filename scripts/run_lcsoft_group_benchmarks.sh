#!/usr/bin/env bash
set -euo pipefail

# Grouped benchmark for the final method as the main subject:
# SASRec / QSDRec semantic-score / Hard CRSID / LC-SoftSID Full.

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASETS="${DATASETS:-runs/beauty runs/sports runs/toys_games}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-256}"
EVAL_BATCH_EVAL_SIZE="${EVAL_BATCH_EVAL_SIZE:-256}"
OUT_NAME="${OUT_NAME:-lcsoft_group_benchmark}"

run_dataset() {
  local dataset_dir="$1"
  local semantic_ids="${SEMANTIC_IDS:-${dataset_dir}/semantic_ids_rq.json}"
  local ablation_root="${ABLATION_ROOT:-${dataset_dir}/lc_soft_required_ablation}"
  local out_dir="${dataset_dir}/${OUT_NAME}"
  local output="${out_dir}/group_benchmark.json"
  local csv="${out_dir}/group_benchmark.csv"
  local checkpoints=()

  mkdir -p "${out_dir}"

  if [[ -f "${ablation_root}/00_sasrec_id_only/best.pt" ]]; then
    checkpoints+=(--checkpoint "sasrec=${ablation_root}/00_sasrec_id_only/best.pt")
  else
    echo "Missing sasrec checkpoint: ${ablation_root}/00_sasrec_id_only/best.pt"
  fi

  if [[ -f "${ablation_root}/01_qsdrec_semantic_score/best.pt" ]]; then
    checkpoints+=(--checkpoint "qsdrec=${ablation_root}/01_qsdrec_semantic_score/best.pt")
  else
    echo "Missing qsdrec checkpoint: ${ablation_root}/01_qsdrec_semantic_score/best.pt"
  fi

  if [[ -f "${ablation_root}/10_hard_crsid/best.pt" ]]; then
    checkpoints+=(--checkpoint "hard_crsid=${ablation_root}/10_hard_crsid/best.pt")
  else
    echo "Missing hard_crsid checkpoint: ${ablation_root}/10_hard_crsid/best.pt"
  fi

  if [[ -f "${ablation_root}/20_lc_soft_full/best.pt" ]]; then
    checkpoints+=(--checkpoint "lcsoft=${ablation_root}/20_lc_soft_full/best.pt")
  else
    echo "Missing lcsoft checkpoint: ${ablation_root}/20_lc_soft_full/best.pt"
  fi

  if [[ "${#checkpoints[@]}" -lt 4 ]]; then
    echo "Warning: fewer than four checkpoints found for ${dataset_dir}. Run required ablations first if needed."
  fi
  if [[ "${#checkpoints[@]}" -eq 0 ]]; then
    echo "Skip ${dataset_dir}: no checkpoints found."
    return
  fi

  echo "============================================================"
  echo "Grouped benchmark: ${dataset_dir}"
  echo "Ablation root:     ${ablation_root}"
  echo "Output:            ${csv}"
  echo "============================================================"

  "${PYTHON_BIN}" scripts/evaluate_lcsoft_group_benchmarks.py \
    --dataset-dir "${dataset_dir}" \
    --semantic-ids "${semantic_ids}" \
    --output "${output}" \
    --csv "${csv}" \
    --main-label lcsoft \
    --device "${DEVICE}" \
    --batch-size "${BATCH_SIZE}" \
    --eval-batch-eval-size "${EVAL_BATCH_EVAL_SIZE}" \
    "${checkpoints[@]}"
}

for dataset_dir in ${DATASETS}; do
  run_dataset "${dataset_dir}"
done
