#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

# Conservative hierarchical-gate rerun. Only the correction bound changes
# from 0.5 to 0.3, so the result is directly comparable with the first probe.
OUTPUT_DIR="${OUTPUT_DIR:-runs/office/hierarchical_residual_gate_scale030_20260614}"
GATE_CORRECTION_SCALE="${GATE_CORRECTION_SCALE:-0.3}"
GATE_KL_WEIGHT="${GATE_KL_WEIGHT:-0.05}"
GATE_PRIVATE_WEIGHT="${GATE_PRIVATE_WEIGHT:-0.1}"
GATE_PRIVATE_MARGIN="${GATE_PRIVATE_MARGIN:-0.05}"
GATE_WARMUP_EPOCHS="${GATE_WARMUP_EPOCHS:-10}"
GATE_LR_SCALE="${GATE_LR_SCALE:-0.1}"

OUTPUT_DIR="${OUTPUT_DIR}" \
GATE_CORRECTION_SCALE="${GATE_CORRECTION_SCALE}" \
GATE_KL_WEIGHT="${GATE_KL_WEIGHT}" \
GATE_PRIVATE_WEIGHT="${GATE_PRIVATE_WEIGHT}" \
GATE_PRIVATE_MARGIN="${GATE_PRIVATE_MARGIN}" \
GATE_WARMUP_EPOCHS="${GATE_WARMUP_EPOCHS}" \
GATE_LR_SCALE="${GATE_LR_SCALE}" \
bash LCSoftCRSID/scripts/run_office_hierarchical_gate.sh

NEW_CHECKPOINT="${OUTPUT_DIR}/best.pt"
FIXED_CHECKPOINT="${FIXED_CHECKPOINT:-runs/office/lcsoft_pfree_probe_20260612/00_prior_guided/best.pt}"
COLD_OUTPUT_DIR="${COLD_OUTPUT_DIR:-${OUTPUT_DIR}/cold_start_comparison}"

if [[ -f "${NEW_CHECKPOINT}" && -f "${FIXED_CHECKPOINT}" ]]; then
  "${PYTHON_BIN:-python}" LCSoftCRSID/scripts/evaluate_cold_start.py \
    --checkpoint "scale030=${NEW_CHECKPOINT}" \
    --checkpoint "fixed=${FIXED_CHECKPOINT}" \
    --output-dir "${COLD_OUTPUT_DIR}" \
    --cold-threshold 5 \
    --device "${DEVICE:-cuda}" \
    --batch-size "${COLD_EVAL_BATCH_SIZE:-128}" \
    --candidate-chunk-size "${COLD_EVAL_CHUNK_SIZE:-512}"
  echo "Cold-start metrics: ${COLD_OUTPUT_DIR}/cold_start_metrics.csv"
else
  echo "Skip cold-start comparison because a checkpoint is missing:"
  echo "  new:   ${NEW_CHECKPOINT}"
  echo "  fixed: ${FIXED_CHECKPOINT}"
fi
