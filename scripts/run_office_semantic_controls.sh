#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET_DIR="${DATASET_DIR:-runs/office}"
SEMANTIC_IDS="${SEMANTIC_IDS:-runs/office/semantic_ids_rq.json}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-runs/office}"
DEVICE="${DEVICE:-cuda}"

EPOCHS="${EPOCHS:-100}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-10}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
MAX_LEN="${MAX_LEN:-50}"
DIM="${DIM:-128}"
NUM_RANDOM_NEG="${NUM_RANDOM_NEG:-100}"

run_exp() {
  local name="$1"
  shift
  local start_ts
  local end_ts
  start_ts="$(date +%s)"
  echo "============================================================"
  echo "Running: ${name}"
  echo "Output: ${BASE_OUTPUT_DIR}/${name}"
  echo "Start:  $(date '+%Y-%m-%d %H:%M:%S')"
  echo "============================================================"
  "${PYTHON_BIN}" scripts/train_qsdrec.py \
    --dataset-dir "${DATASET_DIR}" \
    --semantic-ids "${SEMANTIC_IDS}" \
    --output-dir "${BASE_OUTPUT_DIR}/${name}" \
    --device "${DEVICE}" \
    --epochs "${EPOCHS}" \
    --early-stop-patience "${EARLY_STOP_PATIENCE}" \
    --batch-size "${BATCH_SIZE}" \
    --eval-batch-eval-size 2048 \
    --max-len "${MAX_LEN}" \
    --dim "${DIM}" \
    --num-interests 8 \
    --num-hard-neg 0 \
    --num-random-neg "${NUM_RANDOM_NEG}" \
    --sem-weight 0.10 \
    --dis-weight 0 \
    --div-weight 0 \
    --lr 0.001 \
    --weight-decay 0.0001 \
    --seed 2026 \
    "$@"
  end_ts="$(date +%s)"
  echo "Finished: ${name}, elapsed=$((end_ts - start_ts))s"
  echo
}

# Existing strongest reference: exp_interest8_sem010.

# A. Semantic hub suppression.
run_exp exp_hub_score002_k8_sem010 \
  --hub-score-weight 0.02

run_exp exp_hub_attn005_k8_sem010 \
  --hub-attn-weight 0.05

run_exp exp_hub_loss001_k8_sem010 \
  --hub-loss-weight 0.001

# D. Intent evidence bottleneck / history-overlap evidence gate.
run_exp exp_evidence_f020_k8_sem010 \
  --evidence-gate history_overlap \
  --evidence-floor 0.20

run_exp exp_evidence_f050_k8_sem010 \
  --evidence-gate history_overlap \
  --evidence-floor 0.50

# Contrastive semantic decoding: expert semantic branch minus amateur semantic matcher.
run_exp exp_contrastive005_k8_sem010 \
  --contrastive-alpha 0.05

run_exp exp_contrastive010_k8_sem010 \
  --contrastive-alpha 0.10

# Combined probes. Keep these small first; only expand if a single mechanism is positive.
run_exp exp_evidence_hub_k8_sem010 \
  --evidence-gate history_overlap \
  --evidence-floor 0.20 \
  --hub-attn-weight 0.05

run_exp exp_evidence_contrastive_k8_sem010 \
  --evidence-gate history_overlap \
  --evidence-floor 0.20 \
  --contrastive-alpha 0.05

run_exp exp_semantic_controls_full_k8_sem010 \
  --evidence-gate history_overlap \
  --evidence-floor 0.20 \
  --hub-attn-weight 0.05 \
  --hub-score-weight 0.02 \
  --contrastive-alpha 0.05
