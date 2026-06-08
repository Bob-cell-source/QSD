#!/usr/bin/env bash
set -euo pipefail

# Jointly tune the GRU encoder and LC-SoftCRSID representation. Every search
# decision is based only on validation NDCG@10; test metrics are produced only
# for the final three-seed confirmation. Existing experiment folders are not
# touched.

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET_DIR="${DATASET_DIR:-runs/office}"
SEMANTIC_IDS="${SEMANTIC_IDS:-${DATASET_DIR}/semantic_ids_rq.json}"
DEVICE="${DEVICE:-cuda}"
OUT_ROOT="${OUT_ROOT:-${DATASET_DIR}/gru4rec_lcsoft_joint_search_v3}"
PHASE="${PHASE:-all}" # all | encoder | optimizer | residual | softsid | confirm
FORCE="${FORCE:-0}"
ALLOW_LARGE_CONFIGS="${ALLOW_LARGE_CONFIGS:-0}" # enable dim=256 / batch=512

SEARCH_EPOCHS="${SEARCH_EPOCHS:-60}"
SEARCH_PATIENCE="${SEARCH_PATIENCE:-8}"
FINAL_EPOCHS="${FINAL_EPOCHS:-100}"
FINAL_PATIENCE="${FINAL_PATIENCE:-12}"
EVAL_BATCH_EVAL_SIZE="${EVAL_BATCH_EVAL_SIZE:-128}"
MAX_LEN="${MAX_LEN:-50}"
BASE_SEED="${BASE_SEED:-2026}"

BEST_FIELDS=(
  dim num_layers dropout batch_size num_random_neg lr weight_decay max_len
  cr_tail_tau cr_residual_scale cr_alpha_frequency_transform
  cr_soft_top_m cr_soft_min_overlap_slots cr_soft_min_support
  cr_soft_support_eta cr_soft_hard_token_prior cr_soft_reliability_floor
  cr_soft_max_neighbors
)

run_lcsoft() {
  local output_dir="$1"
  local tune_only="$2"
  local epochs="$3"
  local patience="$4"
  local seed="$5"
  shift 5

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
    --model-variant gru4rec_lcsoft \
    --dataset-dir "${DATASET_DIR}" \
    --semantic-ids "${SEMANTIC_IDS}" \
    --output-dir "${output_dir}" \
    --device "${DEVICE}" \
    --epochs "${epochs}" \
    --early-stop-patience "${patience}" \
    --eval-batch-eval-size "${EVAL_BATCH_EVAL_SIZE}" \
    --max-len "${MAX_LEN}" \
    --num-hard-neg 0 \
    --train-objective sampled \
    --grad-clip 5.0 \
    --seed "${seed}" \
    --sem-weight 1.0 --dis-weight 0 --div-weight 0 \
    --cr-soft-min-overlap-slots 2 \
    --cr-soft-lift-kappa 0.0 \
    --cr-soft-behavior-weight 0.0 \
    "${test_args[@]}" \
    "$@"
}

select_best() {
  eval "$("${PYTHON_BIN}" scripts/select_best_valid_config.py \
    --root "${OUT_ROOT}/search" \
    --variant gru4rec_lcsoft \
    --prefix BEST_ \
    --fields "${BEST_FIELDS[@]}")"
  echo "Selected LC-SoftCRSID: valid=${BEST_VALID_NDCG}"
  echo "Source: ${BEST_RESULT_PATH}"
}

best_args() {
  BEST_ARGS=(
    --dim "${BEST_DIM}"
    --num-layers "${BEST_NUM_LAYERS}"
    --dropout "${BEST_DROPOUT}"
    --batch-size "${BEST_BATCH_SIZE}"
    --num-random-neg "${BEST_NUM_RANDOM_NEG}"
    --lr "${BEST_LR}"
    --weight-decay "${BEST_WEIGHT_DECAY}"
    --cr-tail-tau "${BEST_CR_TAIL_TAU}"
    --cr-residual-scale "${BEST_CR_RESIDUAL_SCALE}"
    --cr-alpha-frequency-transform "${BEST_CR_ALPHA_FREQUENCY_TRANSFORM}"
    --cr-soft-top-m "${BEST_CR_SOFT_TOP_M}"
    --cr-soft-min-overlap-slots "${BEST_CR_SOFT_MIN_OVERLAP_SLOTS}"
    --cr-soft-min-support "${BEST_CR_SOFT_MIN_SUPPORT}"
    --cr-soft-support-eta "${BEST_CR_SOFT_SUPPORT_ETA}"
    --cr-soft-hard-token-prior "${BEST_CR_SOFT_HARD_TOKEN_PRIOR}"
    --cr-soft-reliability-floor "${BEST_CR_SOFT_RELIABILITY_FLOOR}"
    --cr-soft-max-neighbors "${BEST_CR_SOFT_MAX_NEIGHBORS}"
  )
}

base_representation_args() {
  BASE_REPR_ARGS=(
    --cr-tail-tau 20
    --cr-residual-scale 1.0
    --cr-alpha-frequency-transform raw
    --cr-soft-top-m 4
    --cr-soft-min-support 0.05
    --cr-soft-support-eta 2.0
    --cr-soft-hard-token-prior 1.0
    --cr-soft-reliability-floor 0.10
    --cr-soft-max-neighbors 50
  )
}

run_encoder_search() {
  local root="${OUT_ROOT}/search/01_encoder"
  base_representation_args

  run_lcsoft "${root}/e00_d64_l1_dr000" 1 "${SEARCH_EPOCHS}" "${SEARCH_PATIENCE}" "${BASE_SEED}" \
    --dim 64 --num-layers 1 --dropout 0.0 --batch-size 256 --num-random-neg 100 --lr 0.001 --weight-decay 0.0001 "${BASE_REPR_ARGS[@]}"
  run_lcsoft "${root}/e01_d64_l1_dr010" 1 "${SEARCH_EPOCHS}" "${SEARCH_PATIENCE}" "${BASE_SEED}" \
    --dim 64 --num-layers 1 --dropout 0.1 --batch-size 256 --num-random-neg 100 --lr 0.001 --weight-decay 0.0001 "${BASE_REPR_ARGS[@]}"
  run_lcsoft "${root}/e02_d128_l1_dr000" 1 "${SEARCH_EPOCHS}" "${SEARCH_PATIENCE}" "${BASE_SEED}" \
    --dim 128 --num-layers 1 --dropout 0.0 --batch-size 256 --num-random-neg 100 --lr 0.001 --weight-decay 0.0001 "${BASE_REPR_ARGS[@]}"
  run_lcsoft "${root}/e03_d128_l1_dr010" 1 "${SEARCH_EPOCHS}" "${SEARCH_PATIENCE}" "${BASE_SEED}" \
    --dim 128 --num-layers 1 --dropout 0.1 --batch-size 256 --num-random-neg 100 --lr 0.001 --weight-decay 0.0001 "${BASE_REPR_ARGS[@]}"
  run_lcsoft "${root}/e04_d128_l1_dr020" 1 "${SEARCH_EPOCHS}" "${SEARCH_PATIENCE}" "${BASE_SEED}" \
    --dim 128 --num-layers 1 --dropout 0.2 --batch-size 256 --num-random-neg 100 --lr 0.001 --weight-decay 0.0001 "${BASE_REPR_ARGS[@]}"
  run_lcsoft "${root}/e05_d128_l2_dr010" 1 "${SEARCH_EPOCHS}" "${SEARCH_PATIENCE}" "${BASE_SEED}" \
    --dim 128 --num-layers 2 --dropout 0.1 --batch-size 256 --num-random-neg 100 --lr 0.001 --weight-decay 0.0001 "${BASE_REPR_ARGS[@]}"
  run_lcsoft "${root}/e06_d128_l2_dr020" 1 "${SEARCH_EPOCHS}" "${SEARCH_PATIENCE}" "${BASE_SEED}" \
    --dim 128 --num-layers 2 --dropout 0.2 --batch-size 256 --num-random-neg 100 --lr 0.001 --weight-decay 0.0001 "${BASE_REPR_ARGS[@]}"

  if [[ "${ALLOW_LARGE_CONFIGS}" == "1" ]]; then
    run_lcsoft "${root}/e07_d256_l1_dr010" 1 "${SEARCH_EPOCHS}" "${SEARCH_PATIENCE}" "${BASE_SEED}" \
      --dim 256 --num-layers 1 --dropout 0.1 --batch-size 256 --num-random-neg 100 --lr 0.001 --weight-decay 0.0001 "${BASE_REPR_ARGS[@]}"
    run_lcsoft "${root}/e08_d256_l2_dr010" 1 "${SEARCH_EPOCHS}" "${SEARCH_PATIENCE}" "${BASE_SEED}" \
      --dim 256 --num-layers 2 --dropout 0.1 --batch-size 256 --num-random-neg 100 --lr 0.001 --weight-decay 0.0001 "${BASE_REPR_ARGS[@]}"
  fi
  select_best
}

run_optimizer_search() {
  select_best
  best_args
  local root="${OUT_ROOT}/search/02_optimizer"

  # Override only optimizer-related arguments from the inherited LC-SoftCRSID configuration.
  for spec in \
    "o00 128 0.0003 0.0001" \
    "o01 128 0.0005 0.0001" \
    "o02 128 0.0010 0.0001" \
    "o03 256 0.0003 0.0001" \
    "o04 256 0.0005 0.0001" \
    "o05 256 0.0010 0.0001" \
    "o06 256 0.0010 0.0000"; do
    read -r name batch lr wd <<<"${spec}"
    run_lcsoft "${root}/${name}_bs${batch}_lr${lr}_wd${wd}" 1 "${SEARCH_EPOCHS}" "${SEARCH_PATIENCE}" "${BASE_SEED}" \
      "${BEST_ARGS[@]}" --batch-size "${batch}" --lr "${lr}" --weight-decay "${wd}"
  done

  if [[ "${ALLOW_LARGE_CONFIGS}" == "1" ]]; then
    run_lcsoft "${root}/o07_bs512_lr0010" 1 "${SEARCH_EPOCHS}" "${SEARCH_PATIENCE}" "${BASE_SEED}" \
      "${BEST_ARGS[@]}" --batch-size 512 --lr 0.001 --weight-decay 0.0001
  fi
  select_best
}

run_residual_search() {
  select_best
  best_args
  local root="${OUT_ROOT}/search/03_residual"

  # First search the shared/private allocation strength.
  for tau in 1 2 5 10 20 40; do
    run_lcsoft "${root}/r_tau${tau}_s10_raw" 1 "${SEARCH_EPOCHS}" "${SEARCH_PATIENCE}" "${BASE_SEED}" \
      "${BEST_ARGS[@]}" --cr-tail-tau "${tau}" --cr-residual-scale 1.0 --cr-alpha-frequency-transform raw
  done
  select_best
  best_args

  # Then refine residual magnitude and frequency transform around the best tau.
  for scale in 0.5 1.0 1.5 2.0; do
    run_lcsoft "${root}/s_tau${BEST_CR_TAIL_TAU}_scale${scale}_raw" 1 "${SEARCH_EPOCHS}" "${SEARCH_PATIENCE}" "${BASE_SEED}" \
      "${BEST_ARGS[@]}" --cr-residual-scale "${scale}" --cr-alpha-frequency-transform raw
  done
  run_lcsoft "${root}/s_tau${BEST_CR_TAIL_TAU}_scale${BEST_CR_RESIDUAL_SCALE}_log" 1 "${SEARCH_EPOCHS}" "${SEARCH_PATIENCE}" "${BASE_SEED}" \
    "${BEST_ARGS[@]}" --cr-alpha-frequency-transform log
  select_best
}

run_softsid_search() {
  select_best
  best_args
  local root="${OUT_ROOT}/search/04_softsid"

  # Targeted combinations cover under-sharing, over-sharing, and sharpening
  # without turning the search into an uninformative exhaustive grid.
  local specs=(
    "s00 2 0.05 2.0"
    "s01 3 0.05 2.0"
    "s02 4 0.00 2.0"
    "s03 4 0.02 2.0"
    "s04 4 0.05 1.0"
    "s05 4 0.05 1.5"
    "s06 4 0.05 2.0"
    "s07 4 0.05 3.0"
    "s08 4 0.10 2.0"
    "s09 8 0.05 2.0"
    "s10 8 0.10 2.0"
  )
  for spec in "${specs[@]}"; do
    read -r name top_m support eta <<<"${spec}"
    run_lcsoft "${root}/${name}_m${top_m}_sup${support}_eta${eta}" 1 "${SEARCH_EPOCHS}" "${SEARCH_PATIENCE}" "${BASE_SEED}" \
      "${BEST_ARGS[@]}" \
      --cr-soft-top-m "${top_m}" \
      --cr-soft-min-support "${support}" \
      --cr-soft-support-eta "${eta}"
  done
  select_best
  best_args

  # Calibrate hard-token retention and reliability after candidate selection.
  local calibration_specs=(
    "c00 0.5 0.05 50"
    "c01 0.5 0.10 50"
    "c02 1.0 0.05 50"
    "c03 1.0 0.10 20"
    "c04 1.0 0.10 50"
    "c05 1.0 0.10 100"
    "c06 1.0 0.20 50"
    "c07 2.0 0.10 50"
  )
  for spec in "${calibration_specs[@]}"; do
    read -r name prior floor neighbors <<<"${spec}"
    run_lcsoft "${root}/${name}_prior${prior}_rel${floor}_n${neighbors}" 1 "${SEARCH_EPOCHS}" "${SEARCH_PATIENCE}" "${BASE_SEED}" \
      "${BEST_ARGS[@]}" \
      --cr-soft-hard-token-prior "${prior}" \
      --cr-soft-reliability-floor "${floor}" \
      --cr-soft-max-neighbors "${neighbors}"
  done
  select_best

  "${PYTHON_BIN}" scripts/summarize_experiments.py \
    --root "${OUT_ROOT}/search" \
    --metric best_valid_NDCG@10 \
    --csv "${OUT_ROOT}/search/summary_by_valid.csv"
}

run_confirmation() {
  select_best
  best_args
  local root="${OUT_ROOT}/final_3seeds"

  for seed in 2026 2027 2028; do
    run_lcsoft "${root}/gru_lcsoft_best_seed${seed}" 0 "${FINAL_EPOCHS}" "${FINAL_PATIENCE}" "${seed}" \
      "${BEST_ARGS[@]}"
  done

  "${PYTHON_BIN}" scripts/summarize_experiments.py \
    --root "${root}" \
    --metric NDCG@10 \
    --csv "${root}/summary_test.csv"
}

if [[ ! -f "${DATASET_DIR}/sequences.json" || ! -f "${SEMANTIC_IDS}" ]]; then
  echo "Missing dataset or semantic IDs under ${DATASET_DIR}" >&2
  exit 1
fi

case "${PHASE}" in
  all)
    run_encoder_search
    run_optimizer_search
    run_residual_search
    run_softsid_search
    run_confirmation
    ;;
  encoder) run_encoder_search ;;
  optimizer) run_optimizer_search ;;
  residual) run_residual_search ;;
  softsid) run_softsid_search ;;
  confirm) run_confirmation ;;
  *)
    echo "Unknown PHASE=${PHASE}; expected all, encoder, optimizer, residual, softsid, or confirm" >&2
    exit 2
    ;;
esac
