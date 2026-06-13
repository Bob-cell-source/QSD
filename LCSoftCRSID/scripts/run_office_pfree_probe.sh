#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET_DIR="${DATASET_DIR:-runs/office}"
SEMANTIC_IDS="${SEMANTIC_IDS:-${DATASET_DIR}/semantic_ids_rq.json}"
OUT_ROOT="${OUT_ROOT:-${DATASET_DIR}/lcsoft_pfree_probe_20260612}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-100}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-10}"
BATCH_SIZE="${BATCH_SIZE:-256}"
EVAL_CHUNK_SIZE="${EVAL_CHUNK_SIZE:-256}"
SEED="${SEED:-2026}"
FORCE="${FORCE:-0}"

run_one() {
  local name="$1"
  local mode="$2"
  local output_dir="${OUT_ROOT}/${name}"
  if [[ "${FORCE}" != "1" && -f "${output_dir}/test_metrics.json" ]]; then
    echo "Skip existing: ${output_dir}"
    return
  fi

  echo "============================================================"
  echo "Running ${name}"
  echo "============================================================"
  "${PYTHON_BIN}" LCSoftCRSID/train.py \
    --dataset-dir "${DATASET_DIR}" \
    --semantic-ids "${SEMANTIC_IDS}" \
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
    --num-random-negatives 100 \
    --tail-tau 20 \
    --alpha-mode fixed \
    --candidate-weight-mode "${mode}" \
    --soft-top-m 4 \
    --soft-min-overlap-slots 3 \
    --soft-min-support 0.05 \
    --soft-reliability-floor 0.1 \
    --soft-max-neighbors 50 \
    --seed "${SEED}"
}

mkdir -p "${OUT_ROOT}"

# Current method: local support prior plus recommendation-guided reweighting.
run_one "00_prior_guided" "prior_guided"

# Simplified variant: neighborhood Top-M candidates with fully learned weights.
run_one "10_neighborhood_learned_pfree" "neighborhood_learned"

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
        "candidate_weight_mode": obj["args"]["candidate_weight_mode"],
        "path": str(path),
    })

if rows:
    output = root / "summary.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(output)
PY
