#!/usr/bin/env bash
set -euo pipefail

# ID-only control for run_gru_lcsoft_best_cross_dataset.sh. The GRU backbone,
# optimizer, sampling, training schedule, and seeds are kept identical; only
# the LC-SoftCRSID item representation is removed.

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASETS="${DATASETS:-runs/beauty runs/toys_games runs/sports}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-100}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-12}"
BATCH_SIZE="${BATCH_SIZE:-128}"
EVAL_BATCH_EVAL_SIZE="${EVAL_BATCH_EVAL_SIZE:-128}"
SEEDS="${SEEDS:-2026 2027 2028}"
FORCE="${FORCE:-0}"

for dataset_dir in ${DATASETS}; do
  semantic_ids="${dataset_dir}/semantic_ids_rq.json"
  out_root="${dataset_dir}/gru_id_office_best_transfer"

  if [[ ! -f "${dataset_dir}/sequences.json" ]]; then
    echo "Skip missing dataset: ${dataset_dir}/sequences.json"
    continue
  fi
  if [[ ! -f "${semantic_ids}" ]]; then
    echo "Skip missing semantic IDs: ${semantic_ids}"
    continue
  fi

  for seed in ${SEEDS}; do
    output_dir="${out_root}/seed${seed}"
    if [[ "${FORCE}" != "1" && -f "${output_dir}/test_metrics.json" ]]; then
      echo "Skip existing: ${output_dir}"
      continue
    fi

    echo "============================================================"
    echo "Dataset: ${dataset_dir}"
    echo "Seed:    ${seed}"
    echo "Model:   GRU ID-only"
    echo "Output:  ${output_dir}"
    echo "============================================================"

    "${PYTHON_BIN}" scripts/train_qsdrec.py \
      --model-variant gru4rec \
      --dataset-dir "${dataset_dir}" \
      --semantic-ids "${semantic_ids}" \
      --output-dir "${output_dir}" \
      --device "${DEVICE}" \
      --epochs "${EPOCHS}" \
      --early-stop-patience "${EARLY_STOP_PATIENCE}" \
      --batch-size "${BATCH_SIZE}" \
      --eval-batch-eval-size "${EVAL_BATCH_EVAL_SIZE}" \
      --max-len 50 \
      --dim 64 \
      --num-layers 1 \
      --dropout 0.1 \
      --lr 0.0003 \
      --weight-decay 0.0001 \
      --num-random-neg 100 \
      --num-hard-neg 0 \
      --train-objective sampled \
      --grad-clip 5.0 \
      --sem-weight 0.0 \
      --dis-weight 0.0 \
      --div-weight 0.0 \
      --seed "${seed}"
  done

  "${PYTHON_BIN}" scripts/summarize_experiments.py \
    --root "${out_root}" \
    --metric NDCG@10 \
    --csv "${out_root}/summary.csv"
done
