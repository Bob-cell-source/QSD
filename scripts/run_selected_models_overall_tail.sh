#!/usr/bin/env bash
set -euo pipefail

# Some server launchers export an empty or non-integer OpenMP setting.
if [[ ! "${OMP_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
  export OMP_NUM_THREADS=1
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET_DIR="${DATASET_DIR:-runs/office}"
ABLATION_ROOT="${ABLATION_ROOT:-${DATASET_DIR}/lc_soft_required_ablation}"
LOCOREC_CHECKPOINT="${LOCOREC_CHECKPOINT:-${DATASET_DIR}/locorec_loo/best.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATASET_DIR}/selected_models_overall_tail}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-128}"
CANDIDATE_CHUNK_SIZE="${CANDIDATE_CHUNK_SIZE:-512}"

mkdir -p "${OUTPUT_DIR}/base_models" "${OUTPUT_DIR}/locorec"

LOCOREC_EVALUATOR="$("${PYTHON_BIN}" - "${LOCOREC_CHECKPOINT}" <<'PY'
import sys
from pathlib import Path
import torch

checkpoint = Path(sys.argv[1])
state = torch.load(checkpoint, map_location="cpu", weights_only=False)
keys = state.get("model", {}).keys()
if any(key.startswith("item_encoder.soft_correction_gate.") for key in keys):
    print("LoCoRecLOOCorrection/evaluate_cold_start.py")
else:
    print("LoCoRec/evaluate_cold_start.py")
PY
)"

if [[ ! -f "${LOCOREC_EVALUATOR}" ]]; then
  echo "Missing evaluator required by checkpoint: ${LOCOREC_EVALUATOR}" >&2
  exit 1
fi
echo "LoCoRec evaluator: ${LOCOREC_EVALUATOR}"

"${PYTHON_BIN}" LCSoftCRSID/scripts/evaluate_cold_start.py \
  --checkpoint "sasrec_id_only=${ABLATION_ROOT}/00_sasrec_id_only/best.pt" \
  --checkpoint "hard_crsid=${ABLATION_ROOT}/10_hard_crsid/best.pt" \
  --checkpoint "no_shared_residual=${ABLATION_ROOT}/30_no_shared_residual/best.pt" \
  --checkpoint "no_private_residual=${ABLATION_ROOT}/31_no_private_residual/best.pt" \
  --output-dir "${OUTPUT_DIR}/base_models" \
  --device "${DEVICE}" \
  --batch-size "${BATCH_SIZE}" \
  --candidate-chunk-size "${CANDIDATE_CHUNK_SIZE}" \
  --cold-threshold 5

"${PYTHON_BIN}" "${LOCOREC_EVALUATOR}" \
  --checkpoint "${LOCOREC_CHECKPOINT}" \
  --output-dir "${OUTPUT_DIR}/locorec" \
  --device "${DEVICE}" \
  --batch-size "${BATCH_SIZE}" \
  --candidate-chunk-size "${CANDIDATE_CHUNK_SIZE}" \
  --cold-threshold 5

"${PYTHON_BIN}" - "${OUTPUT_DIR}" "${LOCOREC_CHECKPOINT}" <<'PY'
import csv
import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
locorec_checkpoint = sys.argv[2]
base = json.loads((output_dir / "base_models/cold_start_metrics.json").read_text())
locorec = json.loads((output_dir / "locorec/cold_start_metrics.json").read_text())

results = list(base["results"])
results.append({
    "name": "locorec",
    "checkpoint": locorec_checkpoint,
    "groups": locorec["groups"],
})
sasrec = results[0]["groups"]
fields = [
    "model", "checkpoint", "group", "count",
    "NDCG@5", "HR@5", "NDCG@10", "HR@10", "NDCG@20", "HR@20",
    "NDCG@10_gain_over_sasrec", "relative_NDCG@10_gain_over_sasrec",
]
rows = []
for result in results:
    for group in ("overall", "cold_0-5"):
        metrics = result["groups"][group]
        baseline = float(sasrec[group]["NDCG@10"])
        gain = float(metrics["NDCG@10"]) - baseline
        rows.append({
            "model": result["name"],
            "checkpoint": result["checkpoint"],
            "group": group,
            **metrics,
            "NDCG@10_gain_over_sasrec": gain,
            "relative_NDCG@10_gain_over_sasrec": gain / baseline if baseline else 0.0,
        })

path = output_dir / "overall_tail_comparison.csv"
with path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
print(path)
PY
