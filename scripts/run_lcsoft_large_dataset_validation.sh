#!/usr/bin/env bash
set -euo pipefail

# Incremental reviewer validation for Beauty, Sports, and Toys/Games.
#
# Intentionally excluded because they already exist:
# - SASRec / QSDRec / Hard CRSID / full LC-SoftCRSID retraining
# - M=8, no local pruning, eta=1
# - shared/private/basis module removal
# - behavior neighbors, local lift, and multi-seed reruns

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASETS="${DATASETS:-runs/beauty runs/sports runs/toys_games}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-100}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-10}"
BATCH_SIZE="${BATCH_SIZE:-512}"
EVAL_CHUNK_SIZE="${EVAL_CHUNK_SIZE:-1024}"
NUM_RANDOM_NEG="${NUM_RANDOM_NEG:-100}"
SEED="${SEED:-2026}"
FORCE="${FORCE:-0}"

RUN_DIAGNOSTICS="${RUN_DIAGNOSTICS:-1}"
RUN_REVIEWER_TRAINING="${RUN_REVIEWER_TRAINING:-1}"
RUN_LEARNABLE_PROBE="${RUN_LEARNABLE_PROBE:-1}"
RUN_TEXT_KNN="${RUN_TEXT_KNN:-1}"
ALLOW_EXACT_TEXT_KNN="${ALLOW_EXACT_TEXT_KNN:-0}"
EXACT_TEXT_KNN_MAX_ITEMS="${EXACT_TEXT_KNN_MAX_ITEMS:-10000}"

dataset_output_root() {
  local dataset_dir="$1"
  echo "${OUT_ROOT:-${dataset_dir}/lcsoft_incremental_validation_20260610}"
}

existing_main_result() {
  local dataset_dir="$1"
  local candidates=(
    "${dataset_dir}/lc_soft_required_ablation/20_lc_soft_full/test_metrics.json"
    "${dataset_dir}/lc_soft_crsid_ablation/20_lc_soft_crsid_full_m4_eta2/test_metrics.json"
  )
  for path in "${candidates[@]}"; do
    if [[ -f "${path}" ]]; then
      echo "${path}"
      return
    fi
  done
  echo ""
}

existing_main_checkpoint() {
  local result_path="$1"
  if [[ -n "${result_path}" && -f "$(dirname "${result_path}")/best.pt" ]]; then
    echo "$(dirname "${result_path}")/best.pt"
  else
    echo ""
  fi
}

run_qsdrec_exp() {
  local dataset_dir="$1"
  local semantic_ids="$2"
  local out_root="$3"
  local name="$4"
  shift 4
  local output_dir="${out_root}/reviewer_training/${name}"
  if [[ "${FORCE}" != "1" && -f "${output_dir}/test_metrics.json" ]]; then
    echo "Skip existing: ${output_dir}"
    return
  fi
  echo "============================================================"
  echo "Reviewer experiment: ${dataset_dir} / ${name}"
  echo "============================================================"
  "${PYTHON_BIN}" scripts/train_qsdrec.py \
    --dataset-dir "${dataset_dir}" \
    --semantic-ids "${semantic_ids}" \
    --output-dir "${output_dir}" \
    --device "${DEVICE}" \
    --epochs "${EPOCHS}" \
    --early-stop-patience "${EARLY_STOP_PATIENCE}" \
    --batch-size "${BATCH_SIZE}" \
    --eval-batch-eval-size "${EVAL_CHUNK_SIZE}" \
    --max-len 50 \
    --dim 128 \
    --num-heads 2 \
    --num-layers 2 \
    --dropout 0.2 \
    --lr 0.001 \
    --weight-decay 0.0001 \
    --grad-clip 5.0 \
    --num-hard-neg 0 \
    --num-random-neg "${NUM_RANDOM_NEG}" \
    --model-variant crsid_soft \
    --cr-residual-scale 1.0 \
    --cr-alpha-frequency-transform raw \
    --cr-soft-top-m 4 \
    --cr-soft-min-support 0.05 \
    --cr-soft-support-eta 2.0 \
    --cr-soft-hard-token-prior 1.0 \
    --cr-soft-reliability-floor 0.10 \
    --cr-soft-max-neighbors 50 \
    --cr-soft-lift-kappa 0.0 \
    --sem-weight 1.0 \
    --dis-weight 0.2 \
    --div-weight 0.01 \
    --seed "${SEED}" \
    "$@"
}

run_learnable_exp() {
  local dataset_dir="$1"
  local semantic_ids="$2"
  local out_root="$3"
  local name="$4"
  shift 4
  local output_dir="${out_root}/learnable_probe/${name}"
  if [[ "${FORCE}" != "1" && -f "${output_dir}/test_metrics.json" ]]; then
    echo "Skip existing: ${output_dir}"
    return
  fi
  echo "============================================================"
  echo "Learnable experiment: ${dataset_dir} / ${name}"
  echo "============================================================"
  "${PYTHON_BIN}" LCSoftCRSID/train.py \
    --dataset-dir "${dataset_dir}" \
    --semantic-ids "${semantic_ids}" \
    --output-dir "${output_dir}" \
    --device "${DEVICE}" \
    --epochs "${EPOCHS}" \
    --early-stop-patience "${EARLY_STOP_PATIENCE}" \
    --batch-size "${BATCH_SIZE}" \
    --eval-candidate-chunk-size "${EVAL_CHUNK_SIZE}" \
    --max-len 50 \
    --dim 128 \
    --num-heads 2 \
    --num-layers 2 \
    --dropout 0.2 \
    --lr 0.001 \
    --weight-decay 0.0001 \
    --grad-clip 5.0 \
    --num-random-negatives "${NUM_RANDOM_NEG}" \
    --tail-tau 20 \
    --residual-scale 1.0 \
    --frequency-transform raw \
    --soft-top-m 4 \
    --soft-min-overlap-slots 2 \
    --soft-min-support 0.05 \
    --soft-support-eta 2.0 \
    --soft-hard-token-prior 1.0 \
    --soft-reliability-floor 0.10 \
    --soft-max-neighbors 50 \
    --seed "${SEED}" \
    "$@"
}

run_dataset() {
  local dataset_dir="$1"
  local semantic_ids="${dataset_dir}/semantic_ids_rq.json"
  local embeddings="${dataset_dir}/item_text_embeddings.npy"
  local embedding_ids="${dataset_dir}/embedding_item_ids.json"
  local out_root
  out_root="$(dataset_output_root "${dataset_dir}")"
  local reference_result
  reference_result="$(existing_main_result "${dataset_dir}")"
  local reference_checkpoint
  reference_checkpoint="$(existing_main_checkpoint "${reference_result}")"
  local num_items
  num_items="$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["num_items"])' "${dataset_dir}/stats.json" 2>/dev/null || echo 0)"
  local text_knn_available=1
  if [[ "${RUN_TEXT_KNN}" == "1" && "${num_items}" -gt "${EXACT_TEXT_KNN_MAX_ITEMS}" ]]; then
    if ! "${PYTHON_BIN}" -c 'import faiss' >/dev/null 2>&1; then
      if [[ "${ALLOW_EXACT_TEXT_KNN}" != "1" ]]; then
        text_knn_available=0
        echo "Skip text-kNN training for ${dataset_dir}: ${num_items} items and FAISS is unavailable."
        echo "Install faiss-cpu, or set ALLOW_EXACT_TEXT_KNN=1 to use slower chunked exact search."
      fi
    fi
  fi

  if [[ ! -f "${dataset_dir}/sequences.json" || ! -f "${semantic_ids}" ]]; then
    echo "Skip incomplete dataset: ${dataset_dir}"
    return
  fi
  mkdir -p "${out_root}/diagnostics" "${out_root}/reviewer_training" "${out_root}/learnable_probe"
  echo "############################################################"
  echo "Dataset: ${dataset_dir}"
  echo "Existing main result: ${reference_result:-not found}"
  echo "Output root: ${out_root}"
  echo "############################################################"

  if [[ "${RUN_DIAGNOSTICS}" == "1" ]]; then
    if [[ -f "${embeddings}" && -f "${embedding_ids}" ]]; then
      "${PYTHON_BIN}" scripts/analyze_lcsoft_neighbor_evidence.py \
        --dataset-dir "${dataset_dir}" \
        --semantic-ids "${semantic_ids}" \
        --embeddings "${embeddings}" \
        --embedding-item-ids "${embedding_ids}" \
        --output "${out_root}/diagnostics/neighbor_evidence.json" \
        --csv "${out_root}/diagnostics/neighbor_evidence.csv"
    else
      echo "Skip text neighbor diagnostics: ${embeddings} or ${embedding_ids} missing."
    fi

    reliability_args=(
      --dataset-dir "${dataset_dir}"
      --semantic-ids "${semantic_ids}"
      --device "${DEVICE}"
      --batch-size "${BATCH_SIZE}"
      --eval-batch-eval-size "${EVAL_CHUNK_SIZE}"
      --output "${out_root}/diagnostics/reliability.json"
      --csv "${out_root}/diagnostics/reliability.csv"
    )
    if [[ -n "${reference_checkpoint}" ]]; then
      reliability_args+=(--checkpoint "${reference_checkpoint}")
    fi
    "${PYTHON_BIN}" scripts/analyze_lcsoft_reliability.py "${reliability_args[@]}"
  fi

  if [[ "${RUN_REVIEWER_TRAINING}" == "1" ]]; then
    # Generic single-slot/global smoothing baseline. This is not the previous
    # no-pruning experiment: it changes the neighborhood definition itself.
    run_qsdrec_exp "${dataset_dir}" "${semantic_ids}" "${out_root}" \
      "10_single_slot_softening" \
      --cr-tail-tau 20 \
      --cr-soft-min-overlap-slots 1

    if [[ "${RUN_TEXT_KNN}" == "1" && "${text_knn_available}" == "1" && -f "${embeddings}" && -f "${embedding_ids}" ]]; then
      run_qsdrec_exp "${dataset_dir}" "${semantic_ids}" "${out_root}" \
        "11_text_knn_softening" \
        --cr-tail-tau 20 \
        --cr-soft-min-overlap-slots 2 \
        --cr-soft-neighbor-source text_knn \
        --cr-soft-text-embeddings "${embeddings}" \
        --cr-soft-text-item-ids "${embedding_ids}"
    else
      echo "Skip text-kNN training for ${dataset_dir}."
    fi

    # Only missing structural sensitivity points. M=8 and eta=1 are excluded
    # because they were already evaluated in the module ablations.
    run_qsdrec_exp "${dataset_dir}" "${semantic_ids}" "${out_root}" \
      "20_strict_overlap_delta3" \
      --cr-tail-tau 20 \
      --cr-soft-min-overlap-slots 3

    run_qsdrec_exp "${dataset_dir}" "${semantic_ids}" "${out_root}" \
      "21_allocation_tau5" \
      --cr-tail-tau 5 \
      --cr-soft-min-overlap-slots 2

    run_qsdrec_exp "${dataset_dir}" "${semantic_ids}" "${out_root}" \
      "22_allocation_tau80" \
      --cr-tail-tau 80 \
      --cr-soft-min-overlap-slots 2
  fi

  if [[ "${RUN_LEARNABLE_PROBE}" == "1" ]]; then
    # One shared fixed control for both attention and alpha comparisons.
    run_learnable_exp "${dataset_dir}" "${semantic_ids}" "${out_root}" \
      "00_fixed_control" \
      --candidate-weight-mode fixed \
      --alpha-mode fixed \
      --attention-kl-weight 0.0

    run_learnable_exp "${dataset_dir}" "${semantic_ids}" "${out_root}" \
      "10_attention_learned_no_prior" \
      --candidate-weight-mode learned \
      --alpha-mode fixed \
      --attention-kl-weight 0.0

    run_learnable_exp "${dataset_dir}" "${semantic_ids}" "${out_root}" \
      "11_attention_prior_guided" \
      --candidate-weight-mode prior_guided \
      --prior-beta-init 1.0 \
      --alpha-mode fixed \
      --attention-kl-weight 0.0

    run_learnable_exp "${dataset_dir}" "${semantic_ids}" "${out_root}" \
      "12_attention_prior_guided_kl1e3" \
      --candidate-weight-mode prior_guided \
      --prior-beta-init 1.0 \
      --alpha-mode fixed \
      --attention-kl-weight 0.001

    run_learnable_exp "${dataset_dir}" "${semantic_ids}" "${out_root}" \
      "20_learnable_monotonic_alpha" \
      --candidate-weight-mode fixed \
      --alpha-mode learnable_monotonic \
      --attention-kl-weight 0.0

    run_learnable_exp "${dataset_dir}" "${semantic_ids}" "${out_root}" \
      "21_learnable_alpha_prior_attention" \
      --candidate-weight-mode prior_guided \
      --prior-beta-init 1.0 \
      --alpha-mode learnable_monotonic \
      --attention-kl-weight 0.0
  fi

  "${PYTHON_BIN}" scripts/summarize_lcsoft_incremental_validation.py \
    --root "${out_root}" \
    --reference "${reference_result}" \
    --output "${out_root}/summary.csv"
}

for dataset_dir in ${DATASETS}; do
  run_dataset "${dataset_dir}"
done

echo "All incremental LC-SoftCRSID validations finished."
