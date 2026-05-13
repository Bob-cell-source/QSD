# QSD-Rec 轻量实验版

这是一版用于验证当前研究想法的 MVP 实现，目标不是一次性做完整论文系统，而是先跑通核心假设：

- Semantic ID 提供跨物品的语义泛化能力。
- 共享语义前缀或完整 semantic ID 碰撞会带来物品级歧义。
- SASRec 分支保留协同记忆能力。
- 多兴趣语义 query 用于对共享语义候选进行个性化消歧。
- Prefix sibling hard negatives 将“语义歧义”变成显式训练信号。

## 1. 预处理 Amazon 数据

Beauty:

```powershell
C:\msys64\ucrt64\bin\python.exe scripts\preprocess_amazon.py `
  --reviews data\reviews_beauty.json `
  --meta data\meta_Beauty.json `
  --output-dir runs\beauty `
  --min-user-inter 5 `
  --min-item-inter 5
```

Office:

```powershell
C:\msys64\ucrt64\bin\python.exe scripts\preprocess_amazon.py `
  --reviews data\reviews_Office_Products.json `
  --meta data\meta_Office_Products.json `
  --output-dir runs\office `
  --min-user-inter 5 `
  --min-item-inter 5
```

输出文件：

- `sequences.json`：用户交互序列。
- `item_meta.json`：物品文本元信息，字段为 `asin/title/brand/categories/description`。
- `stats.json`：数据规模统计。

说明：

- 预处理后的 `item_meta.json` 不再保存聚合 `text` 字段。
- `description` 保留原始商品描述，避免把 `title/brand/categories` 重复拼接进去。

## 2. 构建 Semantic ID

当前版本默认使用编码器生成 item embedding，再用 RQ-KMeans 构建 semantic ID。

编码器实际输入格式为：

- `Title: ...`
- `Brand: ...`
- `Category: a > b > c`
- `Description: cleaned description`

其中 `Description` 只来自原始 `description` 字段，不再使用重复拼接的聚合 `text`。

Beauty:

```powershell
D:\Users\111\anaconda3\envs\sensevoice\python.exe scripts\build_semantic_ids.py build `
  --item-meta runs\beauty\item_meta.json `
  --output runs\beauty\semantic_ids_rq.json `
  --encoder-model BAAI/bge-small-en-v1.5 `
  --codebook-sizes 64,128,256,512 `
  --batch-size 64 `
  --max-length 256 `
  --save-embeddings runs\beauty\item_text_embeddings.npy `
  --save-item-ids runs\beauty\embedding_item_ids.json
```

Office:

```powershell
D:\Users\111\anaconda3\envs\sensevoice\python.exe scripts\build_semantic_ids.py build `
  --item-meta runs\office\item_meta.json `
  --output runs\office\semantic_ids_rq.json `
  --encoder-model BAAI/bge-small-en-v1.5 `
  --codebook-sizes 64,128,256,512 `
  --batch-size 64 `
  --max-length 512 `
  --save-embeddings runs\office\item_text_embeddings.npy `
  --save-item-ids runs\office\embedding_item_ids.json
```

如果你已经有现成的 item embedding，也可以跳过编码步骤，只做 RQ-KMeans：

```powershell
D:\Users\111\anaconda3\envs\sensevoice\python.exe scripts\build_semantic_ids.py rq-kmeans `
  --embeddings runs\office\item_text_embeddings.npy `
  --item-ids runs\office\embedding_item_ids.json `
  --output runs\office\semantic_ids_rq.json `
  --codebook-sizes 64,128,256,512
```

`--encoder-model` 可以是 Hugging Face 模型名，也可以是本地模型目录。

## 3. 统计碰撞与共享前缀

Beauty:

```powershell
D:\Users\111\anaconda3\envs\sensevoice\python.exe scripts\build_semantic_ids.py analyze `
  --semantic-ids runs\beauty\semantic_ids_rq.json `
  --output runs\beauty\semantic_id_report.json
```

Office:

```powershell
D:\Users\111\anaconda3\envs\sensevoice\python.exe scripts\build_semantic_ids.py analyze `
  --semantic-ids runs\office\semantic_ids_rq.json `
  --output runs\office\semantic_id_report.json
```

重点看 `semantic_id_report.json` 里的几类指标：

- full semantic ID collision rate；
- prefix-1/2/3/4 下的 group size；
- 最大 collision group；
- `groups_gt_1`、`groups_gt_5`、`groups_gt_10`。

如果完整 ID 碰撞较少，但短前缀 group 很大，论文重点应放在 `shared-prefix ambiguity`。

如果完整 ID 碰撞很多，就需要强调协同分支或 `item-level residual` 对 `full-ID collision` 的修正作用。

## 4. 训练 QSD-Rec

训练建议直接用 `sensevoice` 环境：

```powershell
D:\Users\111\anaconda3\envs\sensevoice\python.exe scripts\train_qsdrec.py `
  --dataset-dir runs\beauty `
  --semantic-ids runs\beauty\semantic_ids_rq.json `
  --output-dir runs\beauty\qsdrec `
  --device cuda `
  --epochs 30 `
  --batch-size 256 `
  --max-len 50 `
  --dim 64 `
  --num-interests 4 `
  --prefix-level 2 `
  --num-hard-neg 20 `
  --num-random-neg 100
```
scripts\train_qsdrec.py `
  --dataset-dir runs\office `
  --semantic-ids runs\office\semantic_ids_rq.json `
  --output-dir runs\office\qsdrec `
  --device cuda `
  --epochs 30 `
  --batch-size 256 `
  --max-len 20 `
  --dim 64 `
  --num-interests 3 `
  --prefix-level 2 `
  --num-hard-neg 10 `
  --num-random-neg 50 `
  --weight-decay 1e-5 `
  --dis-weight 0.1 `
  --sem-weight 0.3 `
  --div-weight 1e-3
纯ID版本
scripts\train_qsdrec.py `
  --dataset-dir runs\office `
  --semantic-ids runs\office\semantic_ids_rq.json `
  --output-dir runs\office\sasrec `
  --device cuda `
  --epochs 30 `
  --batch-size 256 `
  --max-len 20 `
  --dim 64 `
  --num-interests 1 `
  --prefix-level 2 `
  --num-hard-neg 0 `
  --num-random-neg 100 `
  --weight-decay 1e-5 `
  --dis-weight 0 `
  --sem-weight 0 `
  --div-weight 1e-3
主要输出：

- `best.pt`：验证集最优模型。
- `history.json`：训练过程。
- `test_metrics.json`：测试指标。

## 5. 建议消融实验

保持主命令不变，每次只改一个因素：

- `--num-interests 1` 对比 `4/6/8`。
- `--num-hard-neg 0`。
- `--prefix-level 1/2/3/4`。
- `--sem-weight 0`，退化为 SASRec-only。
- `--dis-weight 0`，去掉 prefix-aware disambiguation loss。

最关键的证据不只是整体 `NDCG/Recall`，而是模型是否在 `high-sharing semantic-ID groups` 上提升明显。

## 6. 当前实现对应的论文假设

当前代码验证的是下面这条主线：

> Semantic ID 能把物品映射到共享语义空间，从而提升泛化；但共享语义结构会造成候选混淆。因此，需要利用协同记忆和用户多兴趣 query，对共享语义候选进行实例级个性化修正。

对应模型得分为：

\[
s(u,i)=s_{id}(u,i)+\lambda s_{sem}(u,i)
\]

其中：

- \(s_{id}\)：SASRec 协同记忆分支。
- \(s_{sem}\)：多兴趣 query 与 semantic ID token 的语义消歧分支。
- Prefix hard negatives：从共享语义前缀的 sibling items 中采样。

loss变化图
python scripts\plot_history.py `
    --history runs\office\sasrec\history.json `
    --output runs\office\sasrec\history_curve.png