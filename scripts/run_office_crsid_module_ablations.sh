#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/voice/bin/python}"
DATASET_DIR="${DATASET_DIR:-runs/office}"
SEMANTIC_IDS="${SEMANTIC_IDS:-runs/office/semantic_ids_rq.json}"
OUT_ROOT="${OUT_ROOT:-runs/office/crsid_module_ablation}"
DEVICE="${DEVICE:-cuda}"

EPOCHS="${EPOCHS:-30}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-5}"
BATCH_SIZE="${BATCH_SIZE:-256}"
EVAL_BATCH_EVAL_SIZE="${EVAL_BATCH_EVAL_SIZE:-1024}"
MAX_LEN="${MAX_LEN:-50}"
DIM="${DIM:-128}"
NUM_RANDOM_NEG="${NUM_RANDOM_NEG:-100}"
SEED="${SEED:-2026}"
FORCE="${FORCE:-0}"

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

  if [[ "${FORCE}" != "1" && -f "${output_dir}/test_metrics.json" ]]; then
    echo "Skip existing: ${name}"
    return
  fi

  echo "============================================================"
  echo "Running: ${name}"
  echo "Output:  ${output_dir}"
  echo "Start:   $(date '+%Y-%m-%d %H:%M:%S')"
  echo "============================================================"

  "${PYTHON_BIN}" scripts/train_qsdrec.py \
    "${COMMON_ARGS[@]}" \
    --output-dir "${output_dir}" \
    "$@"
}

# Baseline without CRSID modules.
run_exp "00_sasrec_id_only" \
  --model-variant qsdrec \
  --num-interests 1 \
  --sem-weight 0 \
  --dis-weight 0 \
  --div-weight 0

# QSDRec semantic-score baseline, useful for showing that CRSID is not just
# adding a semantic branch.
run_exp "01_qsdrec_semantic_score" \
  --model-variant qsdrec \
  --num-interests 4 \
  --sem-weight 0.10 \
  --dis-weight 0 \
  --div-weight 0

# CRSID full method: semantic basis + adaptive private/shared residual.
run_exp "10_crsid_full_tau20_s10" \
  --model-variant crsid \
  --cr-tail-tau 20 \
  --cr-residual-scale 1.0

# Remove the whole collaborative residual. This leaves only the semantic basis.
run_exp "11_crsid_basis_only_no_residual" \
  --model-variant crsid \
  --cr-tail-tau 20 \
  --cr-residual-scale 0.0

# Remove semantic basis. This tests whether the residual path alone is enough.
run_exp "12_crsid_no_semantic_basis" \
  --model-variant crsid \
  --cr-tail-tau 20 \
  --cr-residual-scale 1.0 \
  --cr-disable-semantic-basis

# Remove shared semantic residual. This tests token-shared collaborative transfer.
run_exp "13_crsid_no_shared_residual" \
  --model-variant crsid \
  --cr-tail-tau 20 \
  --cr-residual-scale 1.0 \
  --cr-disable-shared-residual

# Remove private item residual. This tests item-specific collaborative memorization.
run_exp "14_crsid_no_private_residual" \
  --model-variant crsid \
  --cr-tail-tau 20 \
  --cr-residual-scale 1.0 \
  --cr-disable-private-residual

# Disable adaptive alpha by fixing private/shared residual mixture.
run_exp "15_crsid_fixed_alpha_050" \
  --model-variant crsid \
  --cr-tail-tau 20 \
  --cr-residual-scale 1.0 \
  --cr-alpha-override 0.5

# All-private and all-shared residual endpoints.
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

# Semantic-hub alpha variant. Keep it as a diagnostic alternative to item-frequency alpha.
run_exp "20_crsid_semhub_full_f005_g10_s10" \
  --model-variant crsid_semhub \
  --cr-hub-alpha-floor 0.05 \
  --cr-hub-alpha-gamma 1.0 \
  --cr-residual-scale 1.0

"${PYTHON_BIN}" scripts/summarize_experiments.py \
  --root "${OUT_ROOT}" \
  --top-k 30 \
  --csv "${OUT_ROOT}/summary.csv"

echo "Finished CRSID module ablation. Summary: ${OUT_ROOT}/summary.csv"
