# RTX 5090 Linux Experiment Setup

This guide configures a Linux server with an NVIDIA RTX 5090 for QSD-Rec experiments.

## 1. Basic Checks

Check GPU and driver:

```bash
nvidia-smi
```

You should see an RTX 5090 with about 32 GB GPU memory.

Check CUDA driver compatibility:

```bash
nvidia-smi | grep "CUDA Version"
```

For RTX 5090, use a recent NVIDIA driver and PyTorch with CUDA 12.8 or newer.

## 2. Create Conda Environment

```bash
conda create -n qsdrec5090 python=3.11 -y
conda activate qsdrec5090
```

Install PyTorch with CUDA 12.8:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Install project dependencies:

```bash
cd /path/to/your/qsdrec_project
pip install -r requirements.txt
pip install sentence-transformers transformers tqdm
```

If the server has no Conda, use `venv`:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install sentence-transformers transformers tqdm
```

## 3. Verify PyTorch and RTX 5090

Run:

```bash
python - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("memory GB:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))
PY
```

Expected:

```text
cuda available: True
gpu: NVIDIA GeForce RTX 5090
memory GB: about 32
```

If you see an error mentioning unsupported `sm_120`, reinstall a newer PyTorch CUDA wheel, preferably `cu128` or newer.

## 4. Copy Project Files

Recommended project layout on the server:

```text
qsdrec_project/
  qsdrec/
  scripts/
  requirements.txt
  README_QSDREC.md
  runs/
    office/
```

If you only train and do not rebuild semantic IDs, the minimum files are:

```text
runs/office/sequences.json
runs/office/stats.json
runs/office/semantic_ids_rq.json
```

If you want to reuse or analyze semantic ID construction, also copy:

```text
runs/office/item_meta.json
runs/office/item_text_embeddings.npy
runs/office/embedding_item_ids.json
runs/office/semantic_id_report.json
```

Example upload with `scp` from local machine:

```bash
scp -r qsdrec scripts requirements.txt README_QSDREC.md RTX5090_LINUX_SETUP.md user@server:/path/to/qsdrec_project/
scp -r runs/office user@server:/path/to/qsdrec_project/runs/
```

## 5. Smoke Test

Run a short SASRec-only job first:

```bash
python scripts/train_qsdrec.py \
  --dataset-dir runs/office \
  --semantic-ids runs/office/semantic_ids_rq.json \
  --output-dir runs/office/smoke_sasrec \
  --device cuda \
  --epochs 3 \
  --early-stop-patience 10 \
  --batch-size 512 \
  --eval-batch-eval-size 2048 \
  --max-len 50 \
  --dim 128 \
  --num-interests 1 \
  --num-hard-neg 0 \
  --num-random-neg 100 \
  --sem-weight 0 \
  --dis-weight 0 \
  --div-weight 0
```

If this runs successfully, the environment is usable.

## 6. Recommended RTX 5090 Settings

For pure SASRec:

```bash
--batch-size 512
--eval-batch-eval-size 2048
--dim 128
```

If GPU memory is still low, try:

```bash
--batch-size 1024
--eval-batch-eval-size 4096
```

For QSD-Rec semantic branch:

```bash
--batch-size 512
--eval-batch-eval-size 1024
--dim 128
--num-interests 4
```

If out of memory occurs, reduce in this order:

```bash
--eval-batch-eval-size 512
--batch-size 256
--dim 64
```

## 7. Full SASRec Baseline

```bash
python scripts/train_qsdrec.py \
  --dataset-dir runs/office \
  --semantic-ids runs/office/semantic_ids_rq.json \
  --output-dir runs/office/exp_5090_sasrec \
  --device cuda \
  --epochs 100 \
  --early-stop-patience 10 \
  --batch-size 512 \
  --eval-batch-eval-size 2048 \
  --max-len 50 \
  --dim 128 \
  --num-interests 1 \
  --num-hard-neg 0 \
  --num-random-neg 100 \
  --sem-weight 0 \
  --dis-weight 0 \
  --div-weight 0
```

## 8. Conservative QSD-Rec Run

Start with weak semantic fusion:

```bash
python scripts/train_qsdrec.py \
  --dataset-dir runs/office \
  --semantic-ids runs/office/semantic_ids_rq.json \
  --output-dir runs/office/exp_5090_qsd_sem010 \
  --device cuda \
  --epochs 100 \
  --early-stop-patience 10 \
  --batch-size 512 \
  --eval-batch-eval-size 1024 \
  --max-len 50 \
  --dim 128 \
  --num-interests 4 \
  --num-hard-neg 0 \
  --num-random-neg 100 \
  --sem-weight 0.10 \
  --dis-weight 0 \
  --div-weight 0
```

Then gradually add auxiliary losses:

```bash
--dis-weight 0.02
--dis-weight 0.05
```

Then add prefix hard negatives:

```bash
--prefix-level 2 --num-hard-neg 5
--prefix-level 2 --num-hard-neg 10
```

Avoid starting with:

```bash
--sem-weight 1.0 --dis-weight 0.2 --div-weight 0.01 --num-hard-neg 20
```

This setting is too aggressive for the current model.

## 9. Run Ablation Script

The Linux ablation script is:

```bash
bash scripts/run_office_ablations.sh
```

Recommended RTX 5090 environment variables:

```bash
export PYTHON_BIN=python
export DEVICE=cuda
export EPOCHS=100
export EARLY_STOP_PATIENCE=10
export BATCH_SIZE=512
export DIM=128
export NUM_RANDOM_NEG=100

bash scripts/run_office_ablations.sh
```

For a short script test:

```bash
EPOCHS=3 bash scripts/run_office_ablations.sh
```

Note: the current ablation script uses the training script default `--eval-batch-eval-size 1024`. This is conservative for RTX 5090 and suitable for QSD-Rec. For pure SASRec-only runs, manually using `2048` or `4096` is usually fine.

## 10. Monitor GPU Usage

In another terminal:

```bash
watch -n 1 nvidia-smi
```

Useful signals:

```text
GPU-Util: should be high during training
Memory-Usage: should stay below 32 GB
Power Draw: RTX 5090 can be very power hungry
```

If GPU utilization is low:

- Increase `--batch-size`.
- Make sure data is on local SSD, not a slow network disk.
- Avoid running too many CPU-heavy jobs on the same server.

## 11. Output Files

Each experiment writes:

```text
best.pt
history.json
test_metrics.json
```

`test_metrics.json` contains:

```json
{
  "test": {
    "HR@5": 0.0,
    "Recall@5": 0.0,
    "NDCG@5": 0.0,
    "HR@10": 0.0,
    "Recall@10": 0.0,
    "NDCG@10": 0.0,
    "HR@20": 0.0,
    "Recall@20": 0.0,
    "NDCG@20": 0.0
  },
  "best_valid_NDCG@10": 0.0,
  "args": {}
}
```

This is the main file for later experiment comparison.

## 12. Common Problems

### PyTorch cannot use RTX 5090

Symptom:

```text
sm_120 is not compatible
```

Fix:

```bash
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### CUDA is unavailable

Check:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

If `nvidia-smi` fails, the NVIDIA driver is not installed or not visible in the environment.

### Out of memory

Reduce:

```bash
--eval-batch-eval-size 512
--batch-size 256
```

For QSD-Rec, full-ranking evaluation is usually the first place that hits OOM.

