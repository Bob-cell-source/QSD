#!/usr/bin/env bash
set -euo pipefail

# One-factor-at-a-time sensitivity analysis for the final LoCoRec protocol.
# Unique configurations per dataset:
#   M:     1, 2, 4, 8
#   delta: 1, 2, 3, 4
#   tau:   5, 20, 80
# The default configuration (M=4, delta=3, tau=20) is trained only once.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
DATASETS="${DATASETS:-office beauty sports}"
SEED="${SEED:-2026}"
ROOT_TAG="${ROOT_TAG:-locorec_sensitivity_20260629}"
FORCE="${FORCE:-0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"
EVAL_CHUNK_SIZE="${EVAL_CHUNK_SIZE:-512}"
COLD_THRESHOLD="${COLD_THRESHOLD:-5}"

run_variant() {
  local label="$1"
  local top_m="$2"
  local overlap="$3"
  local tau="$4"

  echo "============================================================"
  echo "Sensitivity: ${label} (M=${top_m}, delta=${overlap}, tau=${tau})"
  echo "============================================================"

  DATASETS="${DATASETS}" \
  SEEDS="${SEED}" \
  RUN_TAG="${ROOT_TAG}/${label}" \
  DEVICE="${DEVICE}" \
  FORCE="${FORCE}" \
  SOFT_TOP_M="${top_m}" \
  SOFT_MIN_OVERLAP_SLOTS="${overlap}" \
  TAIL_TAU="${tau}" \
    bash LCSoftCRSID/scripts/run_locorec_paper.sh
}

# Shared default point.
run_variant default 4 3 20

# Candidate-set size M.
run_variant m_1 1 3 20
run_variant m_2 2 3 20
run_variant m_8 8 3 20

# SID-overlap threshold delta.
run_variant delta_1 4 1 20
run_variant delta_2 4 2 20
run_variant delta_4 4 4 20

# Shared/private allocation strength tau.
run_variant tau_5 4 3 5
run_variant tau_80 4 3 80

result_dirs=()
for dataset in ${DATASETS}; do
  dataset_root="runs/${dataset}/${ROOT_TAG}"
  output_dir="${dataset_root}/evaluation"
  result_dirs+=("${output_dir}")

  checkpoints=(
    --checkpoint "default=${dataset_root}/default/seed${SEED}/best.pt"
    --checkpoint "m_1=${dataset_root}/m_1/seed${SEED}/best.pt"
    --checkpoint "m_2=${dataset_root}/m_2/seed${SEED}/best.pt"
    --checkpoint "m_8=${dataset_root}/m_8/seed${SEED}/best.pt"
    --checkpoint "delta_1=${dataset_root}/delta_1/seed${SEED}/best.pt"
    --checkpoint "delta_2=${dataset_root}/delta_2/seed${SEED}/best.pt"
    --checkpoint "delta_4=${dataset_root}/delta_4/seed${SEED}/best.pt"
    --checkpoint "tau_5=${dataset_root}/tau_5/seed${SEED}/best.pt"
    --checkpoint "tau_80=${dataset_root}/tau_80/seed${SEED}/best.pt"
  )

  "${PYTHON_BIN}" LCSoftCRSID/scripts/evaluate_cold_start.py \
    "${checkpoints[@]}" \
    --output-dir "${output_dir}" \
    --cold-threshold "${COLD_THRESHOLD}" \
    --device "${DEVICE}" \
    --batch-size "${EVAL_BATCH_SIZE}" \
    --candidate-chunk-size "${EVAL_CHUNK_SIZE}"
done

summary_dir="runs/${ROOT_TAG}_summary"
"${PYTHON_BIN}" LCSoftCRSID/scripts/summarize_hyperparameter_sensitivity.py \
  --result-dirs "${result_dirs[@]}" \
  --output-dir "${summary_dir}" \
  --tail-group "cold_0-${COLD_THRESHOLD}"

echo "Sensitivity CSV: ${summary_dir}/summary.csv"
echo "LaTeX table:    ${summary_dir}/table.tex"
echo "Figure:         ${summary_dir}/sensitivity.png"

