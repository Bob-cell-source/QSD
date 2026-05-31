#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET_DIR="${DATASET_DIR:-runs/beauty}"
SEMANTIC_IDS="${SEMANTIC_IDS:-runs/beauty/semantic_ids_rq.json}"
OUT_ROOT="${OUT_ROOT:-runs/beauty/reliability_full}"
DEVICE="${DEVICE:-cuda}"

EPOCHS="${EPOCHS:-100}"
PATIENCE="${PATIENCE:-10}"
BATCH_SIZE="${BATCH_SIZE:-512}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2048}"
MAX_LEN="${MAX_LEN:-50}"
DIM="${DIM:-128}"
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
    "$@"

  end_ts="$(date +%s)"
  elapsed=$((end_ts - start_ts))
  echo "Finished: ${name}"
  echo "End:      $(date '+%Y-%m-%d %H:%M:%S')"
  echo "Elapsed:  $(format_seconds "${elapsed}")"
  echo
}

mkdir -p "${OUT_ROOT}"

# ============================================================
# Stage 1: Core comparison, single seed.
# ============================================================

run_exp seed2026_sasrec \
  --seed 2026 \
  --num-interests 1 \
  --sem-weight 0

run_exp seed2026_qsd_k8_sem010 \
  --seed 2026 \
  --num-interests 8 \
  --sem-weight 0.10

run_exp seed2026_evi_binary_f020_k8_sem010 \
  --seed 2026 \
  --num-interests 8 \
  --sem-weight 0.10 \
  --evidence-gate history_overlap \
  --evidence-floor 0.20

run_exp seed2026_evi_binary_hubpen005_f020_k8_sem010 \
  --seed 2026 \
  --num-interests 8 \
  --sem-weight 0.10 \
  --evidence-gate history_overlap \
  --evidence-floor 0.20 \
  --hub-penalty-weight 0.05

run_exp seed2026_evi_learnable_f020_r100_c020_k8_sem010 \
  --seed 2026 \
  --num-interests 8 \
  --sem-weight 0.10 \
  --evidence-gate learnable \
  --evidence-floor 0.20 \
  --evidence-recency-weight 1.00 \
  --evidence-cross-weight 0.20

# ============================================================
# Stage 2: Anti-overfitting variants for learnable reliability.
# These are useful because valid/test gap is large and the best
# checkpoint often appears around epoch 5-8.
# ============================================================

run_exp seed2026_evi_learnable_f020_r100_c020_k4_sem010 \
  --seed 2026 \
  --num-interests 4 \
  --sem-weight 0.10 \
  --evidence-gate learnable \
  --evidence-floor 0.20 \
  --evidence-recency-weight 1.00 \
  --evidence-cross-weight 0.20

run_exp seed2026_evi_learnable_f020_r100_c020_k8_sem010_drop030 \
  --seed 2026 \
  --num-interests 8 \
  --sem-weight 0.10 \
  --dropout 0.30 \
  --evidence-gate learnable \
  --evidence-floor 0.20 \
  --evidence-recency-weight 1.00 \
  --evidence-cross-weight 0.20

run_exp seed2026_evi_learnable_f020_r100_c020_k8_sem010_wd0005 \
  --seed 2026 \
  --num-interests 8 \
  --sem-weight 0.10 \
  --weight-decay 0.0005 \
  --evidence-gate learnable \
  --evidence-floor 0.20 \
  --evidence-recency-weight 1.00 \
  --evidence-cross-weight 0.20

run_exp seed2026_evi_learnable_f020_r100_c020_k8_sem005 \
  --seed 2026 \
  --num-interests 8 \
  --sem-weight 0.05 \
  --evidence-gate learnable \
  --evidence-floor 0.20 \
  --evidence-recency-weight 1.00 \
  --evidence-cross-weight 0.20

# ============================================================
# Stage 3: Multi-seed stability for the two most important methods.
# Keep this concise: Binary Evidence is the strongest rule baseline,
# Learnable Reliability is the current candidate main method.
# ============================================================

for seed in 2024 2025; do
  run_exp "seed${seed}_evi_binary_f020_k8_sem010" \
    --seed "${seed}" \
    --num-interests 8 \
    --sem-weight 0.10 \
    --evidence-gate history_overlap \
    --evidence-floor 0.20

  run_exp "seed${seed}_evi_learnable_f020_r100_c020_k8_sem010" \
    --seed "${seed}" \
    --num-interests 8 \
    --sem-weight 0.10 \
    --evidence-gate learnable \
    --evidence-floor 0.20 \
    --evidence-recency-weight 1.00 \
    --evidence-cross-weight 0.20
done

# ============================================================
# Summary.
# ============================================================

"${PYTHON_BIN}" scripts/summarize_experiments.py \
  --root "${OUT_ROOT}" \
  --metric NDCG@10 \
  --top-k 100 \
  --csv "${OUT_ROOT}/experiment_summary.csv"

SCRIPT_END_TS="$(date +%s)"
SCRIPT_ELAPSED=$((SCRIPT_END_TS - SCRIPT_START_TS))
echo "============================================================"
echo "All Beauty reliability full experiments finished"
echo "Summary: ${OUT_ROOT}/experiment_summary.csv"
echo "Total elapsed: $(format_seconds "${SCRIPT_ELAPSED}")"
echo "============================================================"

