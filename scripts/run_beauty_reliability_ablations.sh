#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET_DIR="${DATASET_DIR:-runs/beauty}"
SEMANTIC_IDS="${SEMANTIC_IDS:-runs/beauty/semantic_ids_rq.json}"
OUT_ROOT="${OUT_ROOT:-runs/beauty/reliability_ablation}"
DEVICE="${DEVICE:-cuda}"

EPOCHS="${EPOCHS:-100}"
PATIENCE="${PATIENCE:-10}"
BATCH_SIZE="${BATCH_SIZE:-512}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2048}"
MAX_LEN="${MAX_LEN:-50}"
DIM="${DIM:-128}"
SEED="${SEED:-2026}"
NUM_RANDOM_NEG="${NUM_RANDOM_NEG:-100}"

SCRIPT_START_TS="$(date +%s)"

format_seconds() {
  local total="$1"
  local hours=$((total / 3600))
  local minutes=$(((total % 3600) / 60))
  local seconds=$((total % 60))
  printf "%02d:%02d:%02d" "${hours}" "${minutes}" "${seconds}"
}

run_exp() {
  local name="$1"
  shift
  local start_ts
  local end_ts
  local elapsed
  start_ts="$(date +%s)"

  echo "============================================================"
  echo "Running: ${name}"
  echo "Output: ${OUT_ROOT}/${name}"
  echo "Start:  $(date '+%Y-%m-%d %H:%M:%S')"
  echo "============================================================"

  "${PYTHON_BIN}" scripts/train_qsdrec.py \
    --dataset-dir "${DATASET_DIR}" \
    --semantic-ids "${SEMANTIC_IDS}" \
    --output-dir "${OUT_ROOT}/${name}" \
    --device "${DEVICE}" \
    --epochs "${EPOCHS}" \
    --early-stop-patience "${PATIENCE}" \
    --batch-size "${BATCH_SIZE}" \
    --eval-batch-eval-size "${EVAL_BATCH_SIZE}" \
    --max-len "${MAX_LEN}" \
    --dim "${DIM}" \
    --num-random-neg "${NUM_RANDOM_NEG}" \
    --num-hard-neg 0 \
    --dis-weight 0 \
    --div-weight 0 \
    --seed "${SEED}" \
    "$@"

  end_ts="$(date +%s)"
  elapsed=$((end_ts - start_ts))
  echo "Finished: ${name}"
  echo "End:      $(date '+%Y-%m-%d %H:%M:%S')"
  echo "Elapsed:  $(format_seconds "${elapsed}")"
  echo
}

mkdir -p "${OUT_ROOT}"

# 0. Baselines.
run_exp exp_sasrec \
  --num-interests 1 \
  --sem-weight 0

run_exp exp_qsd_k8_sem010 \
  --num-interests 8 \
  --sem-weight 0.10

# 1. Current best rule-based binary evidence.
run_exp exp_evi_binary_f020_k8_sem010 \
  --num-interests 8 \
  --sem-weight 0.10 \
  --evidence-gate history_overlap \
  --evidence-floor 0.20

# 2. Evidence strength: frequency + recency + saturation.
run_exp exp_evi_strength_f020_r100_k8_sem010 \
  --num-interests 8 \
  --sem-weight 0.10 \
  --evidence-gate strength \
  --evidence-floor 0.20 \
  --evidence-recency-weight 1.00

# 3. Evidence strength + token specificity.
run_exp exp_evi_strength_idf_f020_r100_k8_sem010 \
  --num-interests 8 \
  --sem-weight 0.10 \
  --evidence-gate strength_idf \
  --evidence-floor 0.20 \
  --evidence-recency-weight 1.00

# 4. Cross-slot auxiliary evidence. Keep cross weight small because RQ codebooks
# are slot-specific and direct cross-slot equality can introduce false evidence.
run_exp exp_evi_cross_idf_f020_r100_c020_k8_sem010 \
  --num-interests 8 \
  --sem-weight 0.10 \
  --evidence-gate cross_strength_idf \
  --evidence-floor 0.20 \
  --evidence-recency-weight 1.00 \
  --evidence-cross-weight 0.20

# 5. Learnable token reliability estimator.
run_exp exp_evi_learnable_f020_r100_c020_k8_sem010 \
  --num-interests 8 \
  --sem-weight 0.10 \
  --evidence-gate learnable \
  --evidence-floor 0.20 \
  --evidence-recency-weight 1.00 \
  --evidence-cross-weight 0.20

# 6. Semantic hubness penalty.
run_exp exp_evi_binary_hubpen005_f020_k8_sem010 \
  --num-interests 8 \
  --sem-weight 0.10 \
  --evidence-gate history_overlap \
  --evidence-floor 0.20 \
  --hub-penalty-weight 0.05

# 7. Evidence-coverage dynamic fusion.
run_exp exp_evi_binary_ecfusion_f020_lfloor020_k8_sem010 \
  --num-interests 8 \
  --sem-weight 0.10 \
  --evidence-gate history_overlap \
  --evidence-floor 0.20 \
  --semantic-fusion evidence_coverage \
  --fusion-floor 0.20

# 8. Combined candidate. Run this after the isolated ablations so it is not
# mistaken for the source of improvement if the simpler variants already win.
run_exp exp_evi_strength_idf_ecfusion_f020_r100_lfloor020_k8_sem010 \
  --num-interests 8 \
  --sem-weight 0.10 \
  --evidence-gate strength_idf \
  --evidence-floor 0.20 \
  --evidence-recency-weight 1.00 \
  --semantic-fusion evidence_coverage \
  --fusion-floor 0.20

"${PYTHON_BIN}" scripts/summarize_experiments.py \
  --root "${OUT_ROOT}" \
  --metric NDCG@10 \
  --top-k 50 \
  --csv "${OUT_ROOT}/experiment_summary.csv"

SCRIPT_END_TS="$(date +%s)"
SCRIPT_ELAPSED=$((SCRIPT_END_TS - SCRIPT_START_TS))
echo "============================================================"
echo "All reliability ablations finished"
echo "Summary: ${OUT_ROOT}/experiment_summary.csv"
echo "Total elapsed: $(format_seconds "${SCRIPT_ELAPSED}")"
echo "============================================================"

