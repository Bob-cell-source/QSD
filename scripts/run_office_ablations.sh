#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="D:/Users/111/anaconda3/envs/sensevoice/python.exe"
DATASET_DIR="${DATASET_DIR:-runs/office}"
SEMANTIC_IDS="${SEMANTIC_IDS:-runs/office/semantic_ids_rq.json}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-runs/office}"
DEVICE="${DEVICE:-cuda}"

EPOCHS="${EPOCHS:-100}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-10}"
BATCH_SIZE="${BATCH_SIZE:-256}"
MAX_LEN="${MAX_LEN:-50}"
DIM="${DIM:-128}"
NUM_RANDOM_NEG="${NUM_RANDOM_NEG:-100}"

run_exp() {
  local name="$1"
  shift

  echo "============================================================"
  echo "Running: ${name}"
  echo "Output: ${BASE_OUTPUT_DIR}/${name}"
  echo "============================================================"

  "${PYTHON_BIN}" scripts/train_qsdrec.py \
    --dataset-dir "${DATASET_DIR}" \
    --semantic-ids "${SEMANTIC_IDS}" \
    --output-dir "${BASE_OUTPUT_DIR}/${name}" \
    --device "${DEVICE}" \
    --epochs "${EPOCHS}" \
    --early-stop-patience "${EARLY_STOP_PATIENCE}" \
    --batch-size "${BATCH_SIZE}" \
    --max-len "${MAX_LEN}" \
    --dim "${DIM}" \
    --num-random-neg "${NUM_RANDOM_NEG}" \
    "$@"
}

# 1. Pure SASRec baseline.
run_exp exp_sasrec \
  --num-interests 1 \
  --num-hard-neg 0 \
  --sem-weight 0 \
  --dis-weight 0 \
  --div-weight 0

# 2. Semantic fusion weight.
run_exp exp_sem005 \
  --num-interests 4 \
  --num-hard-neg 0 \
  --sem-weight 0.05 \
  --dis-weight 0 \
  --div-weight 0

run_exp exp_sem010 \
  --num-interests 4 \
  --num-hard-neg 0 \
  --sem-weight 0.10 \
  --dis-weight 0 \
  --div-weight 0

run_exp exp_sem020 \
  --num-interests 4 \
  --num-hard-neg 0 \
  --sem-weight 0.20 \
  --dis-weight 0 \
  --div-weight 0

run_exp exp_sem050 \
  --num-interests 4 \
  --num-hard-neg 0 \
  --sem-weight 0.50 \
  --dis-weight 0 \
  --div-weight 0

run_exp exp_sem100 \
  --num-interests 4 \
  --num-hard-neg 0 \
  --sem-weight 1.00 \
  --dis-weight 0 \
  --div-weight 0

# 3. Multi-interest query.
run_exp exp_interest1_sem010 \
  --num-interests 1 \
  --num-hard-neg 0 \
  --sem-weight 0.10 \
  --dis-weight 0 \
  --div-weight 0

run_exp exp_interest2_sem010 \
  --num-interests 2 \
  --num-hard-neg 0 \
  --sem-weight 0.10 \
  --dis-weight 0 \
  --div-weight 0

run_exp exp_interest4_sem010 \
  --num-interests 4 \
  --num-hard-neg 0 \
  --sem-weight 0.10 \
  --dis-weight 0 \
  --div-weight 0

run_exp exp_interest8_sem010 \
  --num-interests 8 \
  --num-hard-neg 0 \
  --sem-weight 0.10 \
  --dis-weight 0 \
  --div-weight 0

# 4. Disambiguation loss.
run_exp exp_dis002 \
  --num-interests 4 \
  --num-hard-neg 0 \
  --sem-weight 0.10 \
  --dis-weight 0.02 \
  --div-weight 0

run_exp exp_dis005 \
  --num-interests 4 \
  --num-hard-neg 0 \
  --sem-weight 0.10 \
  --dis-weight 0.05 \
  --div-weight 0

run_exp exp_dis010 \
  --num-interests 4 \
  --num-hard-neg 0 \
  --sem-weight 0.10 \
  --dis-weight 0.10 \
  --div-weight 0

run_exp exp_dis020 \
  --num-interests 4 \
  --num-hard-neg 0 \
  --sem-weight 0.10 \
  --dis-weight 0.20 \
  --div-weight 0

# 5. Diversity loss.
run_exp exp_div001 \
  --num-interests 4 \
  --num-hard-neg 0 \
  --sem-weight 0.10 \
  --dis-weight 0.05 \
  --div-weight 0.001

run_exp exp_div005 \
  --num-interests 4 \
  --num-hard-neg 0 \
  --sem-weight 0.10 \
  --dis-weight 0.05 \
  --div-weight 0.005

run_exp exp_div010 \
  --num-interests 4 \
  --num-hard-neg 0 \
  --sem-weight 0.10 \
  --dis-weight 0.05 \
  --div-weight 0.010

# 6. Prefix hard negatives.
run_exp exp_hard5_p2 \
  --num-interests 4 \
  --prefix-level 2 \
  --num-hard-neg 5 \
  --sem-weight 0.10 \
  --dis-weight 0.05 \
  --div-weight 0.005

run_exp exp_hard10_p2 \
  --num-interests 4 \
  --prefix-level 2 \
  --num-hard-neg 10 \
  --sem-weight 0.10 \
  --dis-weight 0.05 \
  --div-weight 0.005

run_exp exp_hard20_p2 \
  --num-interests 4 \
  --prefix-level 2 \
  --num-hard-neg 20 \
  --sem-weight 0.10 \
  --dis-weight 0.05 \
  --div-weight 0.005

run_exp exp_hard10_p1 \
  --num-interests 4 \
  --prefix-level 1 \
  --num-hard-neg 10 \
  --sem-weight 0.10 \
  --dis-weight 0.05 \
  --div-weight 0.005

run_exp exp_hard10_p3 \
  --num-interests 4 \
  --prefix-level 3 \
  --num-hard-neg 10 \
  --sem-weight 0.10 \
  --dis-weight 0.05 \
  --div-weight 0.005
