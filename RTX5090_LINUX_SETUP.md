# RTX 5090 Linux 服务器完整实验流程

这份文档用于在 Linux 服务器上从零配置 RTX 5090 实验环境，并完整运行 QSD-Rec 项目，包括数据预处理、Semantic ID 构建、碰撞分析、训练和消融实验。

## 1. 检查服务器 GPU

先确认服务器能看到 RTX 5090：

```bash
nvidia-smi
```

应该能看到类似：

```text
NVIDIA GeForce RTX 5090
Memory: about 32 GB
```

再查看驱动支持的 CUDA 版本：

```bash
nvidia-smi | grep "CUDA Version"
```

RTX 5090 属于 Blackwell 架构，建议使用较新的 NVIDIA 驱动，并安装支持 CUDA 12.8 或更新版本的 PyTorch。

## 2. 创建 Python 环境

推荐使用 Conda：

```bash
conda create -n qsdrec5090 python=3.11 -y
conda activate qsdrec5090
```

安装 PyTorch CUDA 12.8：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

安装项目依赖：

```bash
cd /path/to/qsdrec_project
pip install -r requirements.txt
pip install sentence-transformers transformers tqdm
```

如果服务器没有 Conda，也可以用 `venv`：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install sentence-transformers transformers tqdm
```

## 3. 验证 PyTorch 是否识别 5090

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

期望输出：

```text
cuda available: True
gpu: NVIDIA GeForce RTX 5090
memory GB: about 32
```

如果出现 `sm_120 is not compatible`，说明 PyTorch 版本太旧，重新安装 `cu128` 或更新版本：

```bash
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

## 4. 准备项目目录

推荐服务器目录结构：

```text
qsdrec_project/
  qsdrec/
  scripts/
  data/
  runs/
  requirements.txt
  README_QSDREC.md
  RTX5090_LINUX_SETUP.md
```

从本地上传代码：

```bash
scp -r qsdrec scripts requirements.txt README_QSDREC.md RTX5090_LINUX_SETUP.md user@server:/path/to/qsdrec_project/
```

如果要在服务器上从原始数据开始完整运行，需要上传 Amazon 原始数据到 `data/`：

```text
data/reviews_Office_Products.json
data/meta_Office_Products.json
data/reviews_beauty.json
data/meta_Beauty.json
```

示例：

```bash
scp -r data user@server:/path/to/qsdrec_project/
```

如果原始文件是 `.json.gz`，当前读取代码也支持。

## 5. 从原始数据预处理

进入项目目录：

```bash
cd /path/to/qsdrec_project
conda activate qsdrec5090
```

Office 数据集：

```bash
python scripts/preprocess_amazon.py \
  --reviews data/reviews_Office_Products.json \
  --meta data/meta_Office_Products.json \
  --output-dir runs/office \
  --min-user-inter 5 \
  --min-item-inter 5
```

Beauty 数据集：

```bash
python scripts/preprocess_amazon.py \
  --reviews data/reviews_beauty.json \
  --meta data/meta_Beauty.json \
  --output-dir runs/beauty \
  --min-user-inter 5 \
  --min-item-inter 5
```

预处理输出：

```text
runs/<dataset>/sequences.json
runs/<dataset>/item_meta.json
runs/<dataset>/user2id.json
runs/<dataset>/item2id.json
runs/<dataset>/stats.json
```

检查数据规模：

```bash
cat runs/office/stats.json
```

## 6. 构建 Semantic ID

当前默认流程是：

1. 用文本编码器把 item metadata 编成 embedding。
2. 用 RQ-KMeans 构建多层 semantic ID。

Office：

```bash
python scripts/build_semantic_ids.py build \
  --item-meta runs/office/item_meta.json \
  --output runs/office/semantic_ids_rq.json \
  --encoder-model BAAI/bge-small-en-v1.5 \
  --codebook-sizes 64,128,256,512 \
  --batch-size 128 \
  --max-length 512 \
  --device cuda \
  --save-embeddings runs/office/item_text_embeddings.npy \
  --save-item-ids runs/office/embedding_item_ids.json
```

Beauty：

```bash
python scripts/build_semantic_ids.py build \
  --item-meta runs/beauty/item_meta.json \
  --output runs/beauty/semantic_ids_rq.json \
  --encoder-model BAAI/bge-small-en-v1.5 \
  --codebook-sizes 64,128,256,512 \
  --batch-size 128 \
  --max-length 256 \
  --device cuda \
  --save-embeddings runs/beauty/item_text_embeddings.npy \
  --save-item-ids runs/beauty/embedding_item_ids.json
```

如果服务器不能访问 Hugging Face，可以提前把模型下载到本地目录，然后把 `--encoder-model` 改成本地路径，例如：

```bash
--encoder-model /path/to/models/bge-small-en-v1.5
```

如果已经有 embedding，只重新跑 RQ-KMeans：

```bash
python scripts/build_semantic_ids.py rq-kmeans \
  --embeddings runs/office/item_text_embeddings.npy \
  --item-ids runs/office/embedding_item_ids.json \
  --output runs/office/semantic_ids_rq.json \
  --codebook-sizes 64,128,256,512
```

## 7. 分析 Semantic ID 碰撞

Office：

```bash
python scripts/build_semantic_ids.py analyze \
  --semantic-ids runs/office/semantic_ids_rq.json \
  --output runs/office/semantic_id_report.json
```

Beauty：

```bash
python scripts/build_semantic_ids.py analyze \
  --semantic-ids runs/beauty/semantic_ids_rq.json \
  --output runs/beauty/semantic_id_report.json
```

重点查看：

```text
runs/<dataset>/semantic_id_report.json
```

关注指标：

```text
prefix-1/2/3/4 unique_groups
avg_size
max_size
groups_gt_1
groups_gt_5
collision_item_rate
```

如果短前缀 group 很大，说明 shared-prefix ambiguity 明显，适合支撑 QSD-Rec 的消歧动机。

## 8. 先做 Smoke Test

在完整训练前，先跑 3 个 epoch 确认环境、数据、显存都正常：

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

如果 smoke test 能正常输出 `test_metrics.json`，说明训练链路可用。

## 9. RTX 5090 推荐训练参数

纯 SASRec：

```text
--batch-size 512
--eval-batch-eval-size 2048
--dim 128
```

QSD-Rec 语义分支：

```text
--batch-size 512
--eval-batch-eval-size 1024
--dim 128
--num-interests 4
```

如果显存充足，可以尝试：

```text
--batch-size 1024
--eval-batch-eval-size 2048
```

如果 OOM，优先降低：

```text
--eval-batch-eval-size 512
--batch-size 256
--dim 64
```

## 10. 正式 SASRec Baseline

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

## 11. 保守 QSD-Rec 实验

先只打开较弱的 semantic fusion：

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

然后逐步加入辅助 loss：

```text
--dis-weight 0.02
--dis-weight 0.05
```

再加入 prefix hard negatives：

```text
--prefix-level 2 --num-hard-neg 5
--prefix-level 2 --num-hard-neg 10
```

不建议一开始就使用：

```text
--sem-weight 1.0 --dis-weight 0.2 --div-weight 0.01 --num-hard-neg 20
```

这个组合对当前模型来说太激进，容易压低全局指标。

## 12. 跑完整消融实验

Linux 消融脚本：

```bash
bash scripts/run_office_ablations.sh
```

RTX 5090 推荐环境变量：

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

短测试：

```bash
EPOCHS=3 bash scripts/run_office_ablations.sh
```

注意：当前消融脚本使用训练脚本默认的 `--eval-batch-eval-size 1024`，对 QSD-Rec 比较稳。如果只跑 SASRec，可以手动设置为 `2048` 或 `4096`。

## 13. 监控 GPU

另开一个终端：

```bash
watch -n 1 nvidia-smi
```

重点看：

```text
GPU-Util
Memory-Usage
Power Draw
```

如果 GPU 利用率低：

```text
1. 增大 batch-size。
2. 确认数据在本地 SSD，不在慢速网络盘。
3. 避免同一服务器上同时跑太多 CPU/IO 任务。
```

## 14. 输出文件

每个实验目录会生成：

```text
best.pt
history.json
test_metrics.json
```

其中 `test_metrics.json` 包含：

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

后续汇总实验结果时，优先读取这个文件。

## 15. 汇总实验结果并选择最优参数

项目提供了实验结果汇总脚本：

```text
scripts/summarize_experiments.py
```

它会递归读取指定目录下所有 `test_metrics.json`，按指定指标排序，并输出当前最优实验的完整参数配置。

按 `NDCG@10` 排序：

```bash
python scripts/summarize_experiments.py \
  --root runs/office \
  --metric NDCG@10 \
  --top-k 20
```

同时导出 CSV：

```bash
python scripts/summarize_experiments.py \
  --root runs/office \
  --metric NDCG@10 \
  --top-k 20 \
  --csv runs/office/experiment_summary.csv
```

如果服务器实验输出到了新目录，例如 `runs/office_5090`：

```bash
python scripts/summarize_experiments.py \
  --root runs/office_5090 \
  --metric NDCG@10 \
  --top-k 20 \
  --csv runs/office_5090/experiment_summary.csv
```

也可以按其他指标排序：

```bash
python scripts/summarize_experiments.py \
  --root runs/office \
  --metric HR@10 \
  --top-k 20
```

推荐论文主结果优先按 `NDCG@10` 选择最优参数，同时检查 `HR@10`、`NDCG@20` 是否一致提升。

## 16. 常见问题

### 5090 不被 PyTorch 支持

典型报错：

```text
sm_120 is not compatible
```

解决：

```bash
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### CUDA 不可用

检查：

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

如果 `nvidia-smi` 都失败，说明驱动或服务器 GPU 暴露有问题。

### 显存不足

优先降低：

```bash
--eval-batch-eval-size 512
--batch-size 256
```

QSD-Rec 的 full-ranking evaluation 通常最容易 OOM。

### Hugging Face 下载失败

可以在本地或其他机器提前下载 `BAAI/bge-small-en-v1.5`，上传到服务器，然后使用本地路径：

```bash
--encoder-model /path/to/models/bge-small-en-v1.5
```
