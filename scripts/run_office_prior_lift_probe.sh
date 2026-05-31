#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET_DIR="${DATASET_DIR:-runs/office}"
SEMANTIC_IDS="${SEMANTIC_IDS:-runs/office/semantic_ids_rq.json}"
EMBEDDINGS="${EMBEDDINGS:-runs/office/item_text_embeddings.npy}"
EMBED_ITEM_IDS="${EMBED_ITEM_IDS:-runs/office/embedding_item_ids.json}"
MINI_CLUSTERS="${MINI_CLUSTERS:-runs/office/mini_evidence_clusters.json}"
OUT_ROOT="${OUT_ROOT:-runs/office/prior_lift_probe}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-30}"
PATIENCE="${PATIENCE:-5}"
BATCH_SIZE="${BATCH_SIZE:-256}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2048}"

run_exp() {
  local name="$1"
  shift
  echo "============================================================"
  echo "Running: ${name}"
  echo "Output: ${OUT_ROOT}/${name}"
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
    --max-len 50 \
    --dim 128 \
    --num-random-neg 100 \
    --num-hard-neg 0 \
    --dis-weight 0 \
    --div-weight 0 \
    --seed 2026 \
    "$@"
}

mkdir -p "${OUT_ROOT}"

if [[ ! -f "${MINI_CLUSTERS}" ]]; then
  "${PYTHON_BIN}" scripts/build_mini_evidence_clusters.py \
    --semantic-ids "${SEMANTIC_IDS}" \
    --embeddings "${EMBEDDINGS}" \
    --item-ids "${EMBED_ITEM_IDS}" \
    --output "${MINI_CLUSTERS}" \
    --min-token-size 20 \
    --target-cluster-size 10 \
    --max-clusters-per-token 8 \
    --seed 2026
fi

run_exp exp_sasrec \
  --num-interests 1 \
  --sem-weight 0

run_exp exp_qsd_k8_sem010 \
  --num-interests 8 \
  --sem-weight 0.10

run_exp exp_evi_binary_f020_k8_sem010 \
  --num-interests 8 \
  --sem-weight 0.10 \
  --evidence-gate history_overlap \
  --evidence-floor 0.20

run_exp exp_evi_learnable_f020_r100_c020_k8_sem010 \
  --num-interests 8 \
  --sem-weight 0.10 \
  --evidence-gate learnable \
  --evidence-floor 0.20 \
  --evidence-recency-weight 1.00 \
  --evidence-cross-weight 0.20

run_exp exp_prior_lift_eta025_tau100_k8_sem010 \
  --num-interests 8 \
  --sem-weight 0.10 \
  --evidence-gate prior_lift \
  --evidence-recency-weight 1.00 \
  --prior-lift-alpha 0.10 \
  --prior-lift-tau 1.00 \
  --prior-lift-eta 0.25

run_exp exp_prior_lift_eta050_tau100_k8_sem010 \
  --num-interests 8 \
  --sem-weight 0.10 \
  --evidence-gate prior_lift \
  --evidence-recency-weight 1.00 \
  --prior-lift-alpha 0.10 \
  --prior-lift-tau 1.00 \
  --prior-lift-eta 0.50

run_exp exp_prior_lift_eta100_tau100_k8_sem010 \
  --num-interests 8 \
  --sem-weight 0.10 \
  --evidence-gate prior_lift \
  --evidence-recency-weight 1.00 \
  --prior-lift-alpha 0.10 \
  --prior-lift-tau 1.00 \
  --prior-lift-eta 1.00

run_exp exp_prior_lift_eta050_tau050_k8_sem010 \
  --num-interests 8 \
  --sem-weight 0.10 \
  --evidence-gate prior_lift \
  --evidence-recency-weight 1.00 \
  --prior-lift-alpha 0.10 \
  --prior-lift-tau 0.50 \
  --prior-lift-eta 0.50

run_exp exp_mini_lift_eta050_tau100_k8_sem010 \
  --num-interests 8 \
  --sem-weight 0.10 \
  --evidence-gate mini_lift \
  --mini-clusters "${MINI_CLUSTERS}" \
  --evidence-recency-weight 1.00 \
  --prior-lift-alpha 0.10 \
  --prior-lift-tau 1.00 \
  --prior-lift-eta 0.50

"${PYTHON_BIN}" scripts/summarize_experiments.py \
  --root "${OUT_ROOT}" \
  --metric NDCG@10 \
  --top-k 50 \
  --csv "${OUT_ROOT}/experiment_summary.csv"

echo "Summary: ${OUT_ROOT}/experiment_summary.csv"

