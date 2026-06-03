#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/voice/bin/python}"
DATASET_DIR="${DATASET_DIR:-runs/beauty}"
SEMANTIC_IDS="${SEMANTIC_IDS:-runs/beauty/semantic_ids_rq.json}"
OUT_ROOT="${OUT_ROOT:-runs/beauty/crsid_module_ablation}"
DEVICE="${DEVICE:-cuda}"

# Beauty has many more users/items than Office. CRSID also constructs candidate
# item representations dynamically, so the defaults are deliberately more
# conservative than the generic Beauty QSD scripts.
EPOCHS="${EPOCHS:-100}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-10}"
BATCH_SIZE="${BATCH_SIZE:-256}"
EVAL_BATCH_EVAL_SIZE="${EVAL_BATCH_EVAL_SIZE:-512}"
MAX_LEN="${MAX_LEN:-50}"
DIM="${DIM:-128}"
NUM_RANDOM_NEG="${NUM_RANDOM_NEG:-100}"
SEED="${SEED:-2026}"
FORCE="${FORCE:-0}"
QUICK="${QUICK:-0}"

SCRIPT_START_TS="$(date +%s)"

format_seconds() {
  local total="$1"
  local hours=$((total / 3600))
  local minutes=$(((total % 3600) / 60))
  local seconds=$((total % 60))
  printf "%02d:%02d:%02d" "${hours}" "${minutes}" "${seconds}"
}

COMMON_ARGS=(
  --dataset-dir "${DATASET_DIR}"
  --semantic-ids "${SEMANTIC_IDS}"
  --device "${DEVICE}"
  --epochs "${EPOCHS}"
  --early-stop-patience "${EARLY_STOP_PATIENCE}"
  --batch-size "${BATCH_SIZE}"
  --eval-batch-eval-size "${EVAL_BATCH_EVAL_SIZE}"
  --max-len "${MAX_LEN}"
  --dim "${DIM}"
  --num-hard-neg 0
  --num-random-neg "${NUM_RANDOM_NEG}"
  --seed "${SEED}"
)

run_exp() {
  local name="$1"
  shift
  local output_dir="${OUT_ROOT}/${name}"
  local start_ts
  local end_ts
  local elapsed

  if [[ "${FORCE}" != "1" && -f "${output_dir}/test_metrics.json" ]]; then
    echo "Skip existing: ${name}"
    return
  fi

  mkdir -p "${OUT_ROOT}"
  start_ts="$(date +%s)"

  echo "============================================================"
  echo "Running: ${name}"
  echo "Output:  ${output_dir}"
  echo "Start:   $(date '+%Y-%m-%d %H:%M:%S')"
  echo "============================================================"

  "${PYTHON_BIN}" scripts/train_qsdrec.py \
    "${COMMON_ARGS[@]}" \
    --output-dir "${output_dir}" \
    "$@"

  end_ts="$(date +%s)"
  elapsed=$((end_ts - start_ts))
  echo "Finished: ${name}"
  echo "Elapsed:  $(format_seconds "${elapsed}")"
  echo
}

# 0. Baselines.
run_exp "00_sasrec_id_only" \
  --model-variant qsdrec \
  --num-interests 1 \
  --sem-weight 0 \
  --dis-weight 0 \
  --div-weight 0

run_exp "01_qsdrec_semantic_score" \
  --model-variant qsdrec \
  --num-interests 4 \
  --sem-weight 0.10 \
  --dis-weight 0 \
  --div-weight 0

# 1. Full CRSID.
run_exp "10_crsid_full_tau20_s10" \
  --model-variant crsid \
  --cr-tail-tau 20 \
  --cr-residual-scale 1.0

# 2. Core module ablations.
run_exp "11_crsid_basis_only_no_residual" \
  --model-variant crsid \
  --cr-tail-tau 20 \
  --cr-residual-scale 0.0

run_exp "12_crsid_no_semantic_basis" \
  --model-variant crsid \
  --cr-tail-tau 20 \
  --cr-residual-scale 1.0 \
  --cr-disable-semantic-basis

run_exp "13_crsid_no_shared_residual" \
  --model-variant crsid \
  --cr-tail-tau 20 \
  --cr-residual-scale 1.0 \
  --cr-disable-shared-residual

run_exp "14_crsid_no_private_residual" \
  --model-variant crsid \
  --cr-tail-tau 20 \
  --cr-residual-scale 1.0 \
  --cr-disable-private-residual

run_exp "15_crsid_fixed_alpha_050" \
  --model-variant crsid \
  --cr-tail-tau 20 \
  --cr-residual-scale 1.0 \
  --cr-alpha-override 0.5

if [[ "${QUICK}" != "1" ]]; then
  # Endpoint and semantic-hub diagnostics. These are useful for the final paper
  # table, but QUICK=1 skips them for faster server probes.
  run_exp "16_crsid_private_only_alpha_100" \
    --model-variant crsid \
    --cr-tail-tau 20 \
    --cr-residual-scale 1.0 \
    --cr-alpha-override 1.0

  run_exp "17_crsid_shared_only_alpha_000" \
    --model-variant crsid \
    --cr-tail-tau 20 \
    --cr-residual-scale 1.0 \
    --cr-alpha-override 0.0

  run_exp "20_crsid_semhub_full_f005_g10_s10" \
    --model-variant crsid_semhub \
    --cr-hub-alpha-floor 0.05 \
    --cr-hub-alpha-gamma 1.0 \
    --cr-residual-scale 1.0
fi

"${PYTHON_BIN}" scripts/summarize_experiments.py \
  --root "${OUT_ROOT}" \
  --metric NDCG@10 \
  --top-k 30 \
  --csv "${OUT_ROOT}/summary.csv"

SCRIPT_END_TS="$(date +%s)"
SCRIPT_ELAPSED=$((SCRIPT_END_TS - SCRIPT_START_TS))
echo "============================================================"
echo "Finished Beauty CRSID module ablation"
echo "Summary: ${OUT_ROOT}/summary.csv"
echo "Total elapsed: $(format_seconds "${SCRIPT_ELAPSED}")"
echo "============================================================"
