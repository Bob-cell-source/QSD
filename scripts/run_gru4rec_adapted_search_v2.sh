#!/usr/bin/env bash
set -euo pipefail

# Validation-only staged tuning for the GRU encoder transfer experiment.
# Existing gru4rec_transfer_probe results are never modified.

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET_DIR="${DATASET_DIR:-runs/office}"
SEMANTIC_IDS="${SEMANTIC_IDS:-${DATASET_DIR}/semantic_ids_rq.json}"
DEVICE="${DEVICE:-cuda}"
OUT_ROOT="${OUT_ROOT:-${DATASET_DIR}/gru4rec_adapted_search_v2}"
PHASE="${PHASE:-all}" # all | backbone | lcsoft | confirm
FORCE="${FORCE:-0}"

EPOCHS="${EPOCHS:-80}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-8}"
MAX_LEN="${MAX_LEN:-50}"
EVAL_BATCH_EVAL_SIZE="${EVAL_BATCH_EVAL_SIZE:-128}"
BASE_SEED="${BASE_SEED:-2026}"
ALLOW_LARGE_CONFIGS="${ALLOW_LARGE_CONFIGS:-0}" # dim=256 / batch=512; keep 0 for an 8GB GPU

COMMON_FIELDS=(dim num_layers dropout lr weight_decay batch_size num_random_neg max_len)
LC_FIELDS=(dim num_layers dropout lr weight_decay batch_size num_random_neg max_len cr_tail_tau cr_residual_scale cr_soft_support_eta)

run_exp() {
  local output_dir="$1"
  local tune_only="$2"
  shift 2

  if [[ "${FORCE}" != "1" && -f "${output_dir}/test_metrics.json" ]]; then
    echo "Skip existing: ${output_dir}"
    return
  fi
  mkdir -p "$(dirname "${output_dir}")"

  local test_args=()
  if [[ "${tune_only}" == "1" ]]; then
    test_args+=(--skip-test-evaluation)
  fi

  echo "============================================================"
  echo "Running: ${output_dir}"
  echo "============================================================"
  "${PYTHON_BIN}" scripts/train_qsdrec.py \
    --dataset-dir "${DATASET_DIR}" \
    --semantic-ids "${SEMANTIC_IDS}" \
    --output-dir "${output_dir}" \
    --device "${DEVICE}" \
    --epochs "${EPOCHS}" \
    --early-stop-patience "${EARLY_STOP_PATIENCE}" \
    --eval-batch-eval-size "${EVAL_BATCH_EVAL_SIZE}" \
    --max-len "${MAX_LEN}" \
    --num-hard-neg 0 \
    --train-objective sampled \
    --grad-clip 5.0 \
    --seed "${BASE_SEED}" \
    "${test_args[@]}" \
    "$@"
}

run_id() {
  local output_dir="$1"
  local tune_only="$2"
  shift 2
  run_exp "${output_dir}" "${tune_only}" \
    --model-variant gru4rec \
    --sem-weight 0 --dis-weight 0 --div-weight 0 \
    "$@"
}

run_lcsoft() {
  local output_dir="$1"
  local tune_only="$2"
  shift 2
  run_exp "${output_dir}" "${tune_only}" \
    --model-variant gru4rec_lcsoft \
    --cr-alpha-frequency-transform raw \
    --cr-soft-top-m 4 \
    --cr-soft-min-overlap-slots 2 \
    --cr-soft-min-support 0.05 \
    --cr-soft-hard-token-prior 1.0 \
    --cr-soft-reliability-floor 0.10 \
    --cr-soft-max-neighbors 50 \
    --cr-soft-lift-kappa 0.0 \
    --cr-soft-behavior-weight 0.0 \
    --sem-weight 1.0 --dis-weight 0 --div-weight 0 \
    "$@"
}

select_id() {
  eval "$("${PYTHON_BIN}" scripts/select_best_valid_config.py \
    --root "${OUT_ROOT}/id_search" \
    --variant gru4rec \
    --prefix ID_ \
    --fields "${COMMON_FIELDS[@]}")"
  echo "Selected ID config: valid=${ID_VALID_NDCG}, ${ID_RESULT_PATH}"
}

select_lc() {
  eval "$("${PYTHON_BIN}" scripts/select_best_valid_config.py \
    --root "${OUT_ROOT}/lcsoft_search" \
    --variant gru4rec_lcsoft \
    --prefix LC_ \
    --fields "${LC_FIELDS[@]}")"
  echo "Selected LC-SoftCRSID config: valid=${LC_VALID_NDCG}, ${LC_RESULT_PATH}"
}

run_backbone_search() {
  local root="${OUT_ROOT}/id_search"

  # Stage A: recurrent architecture. One-layer GRUs use external input dropout.
  run_id "${root}/a00_d64_l1_dr000" 1 --dim 64  --num-layers 1 --dropout 0.0 --batch-size 256 --num-random-neg 100 --lr 0.001 --weight-decay 0.0001
  run_id "${root}/a01_d64_l1_dr010" 1 --dim 64  --num-layers 1 --dropout 0.1 --batch-size 256 --num-random-neg 100 --lr 0.001 --weight-decay 0.0001
  run_id "${root}/a02_d128_l1_dr000" 1 --dim 128 --num-layers 1 --dropout 0.0 --batch-size 256 --num-random-neg 100 --lr 0.001 --weight-decay 0.0001
  run_id "${root}/a03_d128_l1_dr010" 1 --dim 128 --num-layers 1 --dropout 0.1 --batch-size 256 --num-random-neg 100 --lr 0.001 --weight-decay 0.0001
  run_id "${root}/a04_d128_l1_dr020" 1 --dim 128 --num-layers 1 --dropout 0.2 --batch-size 256 --num-random-neg 100 --lr 0.001 --weight-decay 0.0001
  run_id "${root}/a05_d128_l2_dr010" 1 --dim 128 --num-layers 2 --dropout 0.1 --batch-size 256 --num-random-neg 100 --lr 0.001 --weight-decay 0.0001
  if [[ "${ALLOW_LARGE_CONFIGS}" == "1" ]]; then
    run_id "${root}/a06_d256_l1_dr010" 1 --dim 256 --num-layers 1 --dropout 0.1 --batch-size 256 --num-random-neg 100 --lr 0.001 --weight-decay 0.0001
    run_id "${root}/a07_d256_l2_dr010" 1 --dim 256 --num-layers 2 --dropout 0.1 --batch-size 256 --num-random-neg 100 --lr 0.001 --weight-decay 0.0001
  fi

  select_id

  # Stage B: optimizer and batch size around the best architecture.
  local arch=(--dim "${ID_DIM}" --num-layers "${ID_NUM_LAYERS}" --dropout "${ID_DROPOUT}" --num-random-neg 100)
  run_id "${root}/b00_bs128_lr0005" 1 "${arch[@]}" --batch-size 128 --lr 0.0005 --weight-decay 0.0001
  run_id "${root}/b01_bs128_lr0010" 1 "${arch[@]}" --batch-size 128 --lr 0.0010 --weight-decay 0.0001
  run_id "${root}/b02_bs256_lr0005" 1 "${arch[@]}" --batch-size 256 --lr 0.0005 --weight-decay 0.0001
  run_id "${root}/b03_bs256_lr0010" 1 "${arch[@]}" --batch-size 256 --lr 0.0010 --weight-decay 0.0001
  if [[ "${ALLOW_LARGE_CONFIGS}" == "1" ]]; then
    run_id "${root}/b04_bs512_lr0010" 1 "${arch[@]}" --batch-size 512 --lr 0.0010 --weight-decay 0.0001
    run_id "${root}/b05_bs512_lr0020" 1 "${arch[@]}" --batch-size 512 --lr 0.0020 --weight-decay 0.0001
  fi

  select_id

  # Stage C: sampled-softmax difficulty under the selected optimizer.
  local opt=(--dim "${ID_DIM}" --num-layers "${ID_NUM_LAYERS}" --dropout "${ID_DROPOUT}" --batch-size "${ID_BATCH_SIZE}" --lr "${ID_LR}" --weight-decay "${ID_WEIGHT_DECAY}")
  run_id "${root}/c00_neg050" 1 "${opt[@]}" --num-random-neg 50
  run_id "${root}/c01_neg100" 1 "${opt[@]}" --num-random-neg 100
  run_id "${root}/c02_neg300" 1 "${opt[@]}" --num-random-neg 300

  select_id
  "${PYTHON_BIN}" scripts/summarize_experiments.py --root "${root}" --metric best_valid_NDCG@10 --csv "${root}/summary_by_valid.csv"
}

run_lcsoft_search() {
  select_id
  local root="${OUT_ROOT}/lcsoft_search"
  local backbone=(
    --dim "${ID_DIM}" --num-layers "${ID_NUM_LAYERS}" --dropout "${ID_DROPOUT}"
    --batch-size "${ID_BATCH_SIZE}" --num-random-neg "${ID_NUM_RANDOM_NEG}"
    --lr "${ID_LR}" --weight-decay "${ID_WEIGHT_DECAY}"
  )

  # Smaller tau preserves more item-private information, which is often needed
  # after replacing self-attention with a recurrent state bottleneck.
  run_lcsoft "${root}/l00_tau1_s10"  1 "${backbone[@]}" --cr-tail-tau 1  --cr-residual-scale 1.0 --cr-soft-support-eta 2.0
  run_lcsoft "${root}/l01_tau2_s10"  1 "${backbone[@]}" --cr-tail-tau 2  --cr-residual-scale 1.0 --cr-soft-support-eta 2.0
  run_lcsoft "${root}/l02_tau5_s10"  1 "${backbone[@]}" --cr-tail-tau 5  --cr-residual-scale 1.0 --cr-soft-support-eta 2.0
  run_lcsoft "${root}/l03_tau10_s10" 1 "${backbone[@]}" --cr-tail-tau 10 --cr-residual-scale 1.0 --cr-soft-support-eta 2.0
  run_lcsoft "${root}/l04_tau20_s10" 1 "${backbone[@]}" --cr-tail-tau 20 --cr-residual-scale 1.0 --cr-soft-support-eta 2.0
  run_lcsoft "${root}/l05_tau2_s15"  1 "${backbone[@]}" --cr-tail-tau 2  --cr-residual-scale 1.5 --cr-soft-support-eta 2.0
  run_lcsoft "${root}/l06_tau5_s15"  1 "${backbone[@]}" --cr-tail-tau 5  --cr-residual-scale 1.5 --cr-soft-support-eta 2.0
  run_lcsoft "${root}/l07_tau10_s15" 1 "${backbone[@]}" --cr-tail-tau 10 --cr-residual-scale 1.5 --cr-soft-support-eta 2.0
  run_lcsoft "${root}/l08_tau2_s20"  1 "${backbone[@]}" --cr-tail-tau 2  --cr-residual-scale 2.0 --cr-soft-support-eta 2.0
  run_lcsoft "${root}/l09_tau5_s20"  1 "${backbone[@]}" --cr-tail-tau 5  --cr-residual-scale 2.0 --cr-soft-support-eta 2.0
  run_lcsoft "${root}/l10_tau2_s15_eta1" 1 "${backbone[@]}" --cr-tail-tau 2 --cr-residual-scale 1.5 --cr-soft-support-eta 1.0
  run_lcsoft "${root}/l11_tau5_s15_eta1" 1 "${backbone[@]}" --cr-tail-tau 5 --cr-residual-scale 1.5 --cr-soft-support-eta 1.0

  select_lc
  "${PYTHON_BIN}" scripts/summarize_experiments.py --root "${root}" --metric best_valid_NDCG@10 --csv "${root}/summary_by_valid.csv"
}

run_confirmation() {
  select_id
  select_lc
  local root="${OUT_ROOT}/final_3seeds"

  for seed in 2026 2027 2028; do
    BASE_SEED="${seed}"
    run_id "${root}/gru_id_seed${seed}" 0 \
      --dim "${ID_DIM}" --num-layers "${ID_NUM_LAYERS}" --dropout "${ID_DROPOUT}" \
      --batch-size "${ID_BATCH_SIZE}" --num-random-neg "${ID_NUM_RANDOM_NEG}" \
      --lr "${ID_LR}" --weight-decay "${ID_WEIGHT_DECAY}"

    run_lcsoft "${root}/gru_lcsoft_seed${seed}" 0 \
      --dim "${LC_DIM}" --num-layers "${LC_NUM_LAYERS}" --dropout "${LC_DROPOUT}" \
      --batch-size "${LC_BATCH_SIZE}" --num-random-neg "${LC_NUM_RANDOM_NEG}" \
      --lr "${LC_LR}" --weight-decay "${LC_WEIGHT_DECAY}" \
      --cr-tail-tau "${LC_CR_TAIL_TAU}" \
      --cr-residual-scale "${LC_CR_RESIDUAL_SCALE}" \
      --cr-soft-support-eta "${LC_CR_SOFT_SUPPORT_ETA}"
  done

  "${PYTHON_BIN}" scripts/summarize_experiments.py --root "${root}" --metric NDCG@10 --csv "${root}/summary_test.csv"
}

if [[ ! -f "${DATASET_DIR}/sequences.json" || ! -f "${SEMANTIC_IDS}" ]]; then
  echo "Missing dataset or semantic IDs under ${DATASET_DIR}" >&2
  exit 1
fi

case "${PHASE}" in
  all)
    run_backbone_search
    run_lcsoft_search
    run_confirmation
    ;;
  backbone) run_backbone_search ;;
  lcsoft) run_lcsoft_search ;;
  confirm) run_confirmation ;;
  *) echo "Unknown PHASE=${PHASE}; expected all, backbone, lcsoft, or confirm" >&2; exit 2 ;;
esac
