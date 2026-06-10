#!/usr/bin/env bash
set -euo pipefail

# Reviewer-facing evidence for LC-SoftCRSID.
# This script intentionally excludes multi-seed reruns.

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET_DIR="${DATASET_DIR:-runs/office}"
SEMANTIC_IDS="${SEMANTIC_IDS:-${DATASET_DIR}/semantic_ids_rq.json}"
EMBEDDINGS="${EMBEDDINGS:-${DATASET_DIR}/item_text_embeddings.npy}"
EMBEDDING_ITEM_IDS="${EMBEDDING_ITEM_IDS:-${DATASET_DIR}/embedding_item_ids.json}"
OUT_ROOT="${OUT_ROOT:-${DATASET_DIR}/reviewer_evidence_20260610}"
DEVICE="${DEVICE:-cuda}"

EPOCHS="${EPOCHS:-100}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-10}"
BATCH_SIZE="${BATCH_SIZE:-512}"
EVAL_BATCH_EVAL_SIZE="${EVAL_BATCH_EVAL_SIZE:-512}"
MAX_LEN="${MAX_LEN:-50}"
DIM="${DIM:-128}"
NUM_RANDOM_NEG="${NUM_RANDOM_NEG:-100}"
SEED="${SEED:-2026}"
FORCE="${FORCE:-0}"
RUN_DIAGNOSTICS="${RUN_DIAGNOSTICS:-1}"
RUN_TRAINING="${RUN_TRAINING:-1}"

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
  --lr 0.001
  --weight-decay 0.0001
  --seed "${SEED}"
  --model-variant crsid_soft
  --cr-residual-scale 1.0
  --cr-alpha-frequency-transform raw
  --cr-soft-min-support 0.05
  --cr-soft-support-eta 2.0
  --cr-soft-hard-token-prior 1.0
  --cr-soft-reliability-floor 0.10
  --cr-soft-max-neighbors 50
  --cr-soft-lift-kappa 0.0
  --sem-weight 1.0
  --dis-weight 0.2
  --div-weight 0.01
)

run_exp() {
  local name="$1"
  shift
  local output_dir="${OUT_ROOT}/training/${name}"
  if [[ "${FORCE}" != "1" && -f "${output_dir}/test_metrics.json" ]]; then
    echo "Skip existing: ${output_dir}"
    return
  fi
  mkdir -p "${output_dir}"
  echo "============================================================"
  echo "Running ${name}"
  echo "Output: ${output_dir}"
  echo "============================================================"
  "${PYTHON_BIN}" scripts/train_qsdrec.py \
    "${COMMON_ARGS[@]}" \
    --output-dir "${output_dir}" \
    "$@"
}

mkdir -p "${OUT_ROOT}/diagnostics" "${OUT_ROOT}/training"

if [[ "${RUN_DIAGNOSTICS}" == "1" ]]; then
  if [[ -f "${EMBEDDINGS}" && -f "${EMBEDDING_ITEM_IDS}" ]]; then
    "${PYTHON_BIN}" scripts/analyze_lcsoft_neighbor_evidence.py \
      --dataset-dir "${DATASET_DIR}" \
      --semantic-ids "${SEMANTIC_IDS}" \
      --embeddings "${EMBEDDINGS}" \
      --embedding-item-ids "${EMBEDDING_ITEM_IDS}" \
      --output "${OUT_ROOT}/diagnostics/neighbor_evidence.json" \
      --csv "${OUT_ROOT}/diagnostics/neighbor_evidence.csv"
  else
    echo "Skip neighbor evidence: text embedding files are missing."
  fi

  "${PYTHON_BIN}" scripts/analyze_lcsoft_reliability.py \
    --dataset-dir "${DATASET_DIR}" \
    --semantic-ids "${SEMANTIC_IDS}" \
    --output "${OUT_ROOT}/diagnostics/reliability_structure.json" \
    --csv "${OUT_ROOT}/diagnostics/reliability_structure.csv"
fi

if [[ "${RUN_TRAINING}" == "1" ]]; then
  # Same formal configuration as the current main method.
  run_exp "00_lcsoft_reference_m4_delta2_tau20" \
    --cr-tail-tau 20 \
    --cr-soft-top-m 4 \
    --cr-soft-min-overlap-slots 2

  # Alternative softening baselines.
  run_exp "10_single_slot_global_softening" \
    --cr-tail-tau 20 \
    --cr-soft-top-m 4 \
    --cr-soft-min-overlap-slots 1

  if [[ -f "${EMBEDDINGS}" && -f "${EMBEDDING_ITEM_IDS}" ]]; then
    run_exp "11_text_knn_softening" \
      --cr-tail-tau 20 \
      --cr-soft-top-m 4 \
      --cr-soft-min-overlap-slots 2 \
      --cr-soft-neighbor-source text_knn \
      --cr-soft-text-embeddings "${EMBEDDINGS}" \
      --cr-soft-text-item-ids "${EMBEDDING_ITEM_IDS}"
  else
    echo "Skip text-kNN softening: text embedding files are missing."
  fi

  # Compact sensitivity analysis: only the three structural parameters.
  run_exp "20_sensitivity_m2" \
    --cr-tail-tau 20 \
    --cr-soft-top-m 2 \
    --cr-soft-min-overlap-slots 2

  run_exp "21_sensitivity_m8" \
    --cr-tail-tau 20 \
    --cr-soft-top-m 8 \
    --cr-soft-min-overlap-slots 2

  run_exp "22_sensitivity_delta3" \
    --cr-tail-tau 20 \
    --cr-soft-top-m 4 \
    --cr-soft-min-overlap-slots 3

  run_exp "23_sensitivity_tau5" \
    --cr-tail-tau 5 \
    --cr-soft-top-m 4 \
    --cr-soft-min-overlap-slots 2

  run_exp "24_sensitivity_tau80" \
    --cr-tail-tau 80 \
    --cr-soft-top-m 4 \
    --cr-soft-min-overlap-slots 2

  "${PYTHON_BIN}" scripts/summarize_experiments.py \
    --root "${OUT_ROOT}/training" \
    --metric NDCG@10 \
    --top-k 30 \
    --csv "${OUT_ROOT}/training/summary.csv"

  reference_checkpoint="${OUT_ROOT}/training/00_lcsoft_reference_m4_delta2_tau20/best.pt"
  if [[ "${RUN_DIAGNOSTICS}" == "1" && -f "${reference_checkpoint}" ]]; then
    "${PYTHON_BIN}" scripts/analyze_lcsoft_reliability.py \
      --dataset-dir "${DATASET_DIR}" \
      --semantic-ids "${SEMANTIC_IDS}" \
      --checkpoint "${reference_checkpoint}" \
      --device "${DEVICE}" \
      --batch-size "${BATCH_SIZE}" \
      --eval-batch-eval-size "${EVAL_BATCH_EVAL_SIZE}" \
      --output "${OUT_ROOT}/diagnostics/reliability_with_performance.json" \
      --csv "${OUT_ROOT}/diagnostics/reliability_with_performance.csv"
  fi
fi

echo "Reviewer evidence finished: ${OUT_ROOT}"
