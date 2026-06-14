#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET_DIR="${DATASET_DIR:-runs/office}"
OUT_ROOT="${OUT_ROOT:-${DATASET_DIR}/three_way_gate_probe_20260614}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-100}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-10}"
BATCH_SIZE="${BATCH_SIZE:-256}"
EVAL_CHUNK_SIZE="${EVAL_CHUNK_SIZE:-256}"
GATE_KL_WEIGHT="${GATE_KL_WEIGHT:-0.001}"
SEED="${SEED:-2026}"
FORCE="${FORCE:-0}"

run_one() {
  local name="$1"
  local fusion_mode="$2"
  local gate_kl_weight="$3"
  local output_dir="${OUT_ROOT}/${name}"
  if [[ "${FORCE}" != "1" && -f "${output_dir}/test_metrics.json" ]]; then
    echo "Skip existing: ${output_dir}"
    return
  fi

  "${PYTHON_BIN}" LCSoftCRSID/train.py \
    --dataset-dir "${DATASET_DIR}" \
    --semantic-ids "${DATASET_DIR}/semantic_ids_rq.json" \
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
    --soft-neighbor-source sid_overlap \
    --soft-top-m 4 \
    --soft-min-overlap-slots 3 \
    --soft-min-support 0.05 \
    --soft-reliability-floor 0.10 \
    --soft-max-neighbors 50 \
    --candidate-weight-mode prior_guided \
    --alpha-mode fixed \
    --fusion-mode "${fusion_mode}" \
    --gate-kl-weight "${gate_kl_weight}" \
    --seed "${SEED}"
}

mkdir -p "${OUT_ROOT}"

# Re-run the control in the same code path for a strict comparison.
run_one "00_fixed_fusion_control" fixed 0.0
run_one "10_prior_guided_three_way_gate" prior_guided_gate "${GATE_KL_WEIGHT}"

"${PYTHON_BIN}" - "${OUT_ROOT}" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for path in sorted(root.glob("*/test_metrics.json")):
    data = json.loads(path.read_text(encoding="utf-8"))
    test = data["test"]
    rows.append({
        "experiment": path.parent.name,
        "best_valid_NDCG@10": data["best_valid_NDCG@10"],
        "NDCG@5": test["NDCG@5"],
        "HR@5": test["HR@5"],
        "NDCG@10": test["NDCG@10"],
        "HR@10": test["HR@10"],
        "NDCG@20": test["NDCG@20"],
        "HR@20": test["HR@20"],
        "fusion_mode": data["args"].get("fusion_mode", "fixed"),
        "gate_kl_weight": data["args"].get("gate_kl_weight", 0.0),
        "gate_statistics": json.dumps(data.get("learned_gate_statistics")),
        "path": str(path),
    })
if rows:
    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(root / "summary.csv")
PY
