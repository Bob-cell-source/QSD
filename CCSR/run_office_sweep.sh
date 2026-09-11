#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
output_root="${OUTPUT_ROOT:-runs/office/ccsr_sweep}"
mkdir -p "$output_root"
for coefficient in 0 0.01 0.03 0.1 0.3; do
  python -m CCSR.train \
    --dataset-dir runs/office \
    --semantic-ids runs/office/semantic_ids_rq.json \
    --init-checkpoint runs/office/locorec_loo_delta2_hard_m1_20260614/best.pt \
    --output-dir "$output_root/lambda_$coefficient" \
    --lambda-res "$coefficient" \
    --device "${DEVICE:-cuda}" \
    --epochs "${EPOCHS:-100}" \
    --seed "${SEED:-2026}" \
    > "$output_root/lambda_$coefficient.log" 2>&1
done
python - "$output_root" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for coefficient in ("0", "0.01", "0.03", "0.1", "0.3"):
    directory = root / f"lambda_{coefficient}"
    result = json.loads((directory / "test_metrics.json").read_text())
    rows.append({"lambda_res": float(coefficient), "output_dir": str(directory),
                 "valid_hard_NDCG@10": result["best_valid_NDCG@10"],
                 "best_epoch": result["best_epoch"]})
best = max(rows, key=lambda row: row["valid_hard_NDCG@10"])
selection = {"selection_criterion": "valid_hard_NDCG@10", "selected": best, "runs": rows}
(root / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
print(json.dumps(selection, indent=2))
PY
