#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET_DIR="${DATASET_DIR:-runs/office}"
SEMANTIC_IDS="${SEMANTIC_IDS:-${DATASET_DIR}/semantic_ids_rq.json}"
TEXT_EMBEDDINGS="${TEXT_EMBEDDINGS:-${DATASET_DIR}/item_text_embeddings.npy}"
TEXT_ITEM_IDS="${TEXT_ITEM_IDS:-${DATASET_DIR}/embedding_item_ids.json}"
OUT_ROOT="${OUT_ROOT:-${DATASET_DIR}/lcsoft_text_knn_comparison_fixed_alpha_20260611}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-256}"
EVAL_CHUNK_SIZE="${EVAL_CHUNK_SIZE:-256}"
FORCE="${FORCE:-0}"

run_one() {
  local name="$1"
  shift
  local output_dir="${OUT_ROOT}/${name}"
  if [[ "${FORCE}" != "1" && -f "${output_dir}/test_metrics.json" ]]; then
    echo "Skip existing: ${output_dir}"
    return
  fi
  "${PYTHON_BIN}" LCSoftCRSID/train.py \
    --dataset-dir "${DATASET_DIR}" \
    --semantic-ids "${SEMANTIC_IDS}" \
    --output-dir "${output_dir}" \
    --device "${DEVICE}" \
    --epochs "${EPOCHS}" \
    --early-stop-patience 10 \
    --batch-size "${BATCH_SIZE}" \
    --eval-candidate-chunk-size "${EVAL_CHUNK_SIZE}" \
    --max-len 50 \
    --dim 128 \
    --num-heads 2 \
    --num-layers 2 \
    --dropout 0.2 \
    --lr 0.001 \
    --weight-decay 0.0001 \
    --num-random-negatives 100 \
    --tail-tau 20 \
    --soft-top-m 4 \
    --soft-min-overlap-slots 3 \
    --soft-min-support 0.05 \
    --soft-reliability-floor 0.1 \
    --soft-max-neighbors 50 \
    --candidate-weight-mode prior_guided \
    --alpha-mode fixed \
    --seed 2026 \
    "$@"
}

if [[ ! -f "${TEXT_EMBEDDINGS}" || ! -f "${TEXT_ITEM_IDS}" ]]; then
  echo "Missing text embedding files:"
  echo "  ${TEXT_EMBEDDINGS}"
  echo "  ${TEXT_ITEM_IDS}"
  exit 1
fi

mkdir -p "${OUT_ROOT}"

run_one "00_sid_overlap_delta3" \
  --soft-neighbor-source sid_overlap

run_one "10_text_knn" \
  --soft-neighbor-source text_knn \
  --soft-text-embeddings "${TEXT_EMBEDDINGS}" \
  --soft-text-item-ids "${TEXT_ITEM_IDS}"

"${PYTHON_BIN}" - "${OUT_ROOT}" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for path in sorted(root.glob("*/test_metrics.json")):
    obj = json.loads(path.read_text(encoding="utf-8"))
    test = obj["test"]
    rows.append({
        "experiment": path.parent.name,
        "best_valid_NDCG@10": obj["best_valid_NDCG@10"],
        "NDCG@5": test["NDCG@5"],
        "HR@5": test["HR@5"],
        "NDCG@10": test["NDCG@10"],
        "HR@10": test["HR@10"],
        "NDCG@20": test["NDCG@20"],
        "HR@20": test["HR@20"],
    })
if rows:
    output = root / "summary.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(output)
PY
