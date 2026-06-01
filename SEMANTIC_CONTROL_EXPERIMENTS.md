# Semantic Control 实验版本说明

本文档记录本轮方法改进的版本管理。所有新增机制默认关闭，旧实验配置不受影响。

## 1. 当前已实现版本

### V0: QSDRec Reference

已有最优参考：

```text
runs/office/exp_interest8_sem010
```

核心配置：

```text
num_interests=8
sem_weight=0.10
num_hard_neg=0
```

Office test:

```text
NDCG@10=0.063522
HR@10=0.112130
```

### V1: Semantic Hub Suppression

对应想法：方向 A，Semantic Hub Suppression。

实现位置：

```text
qsdrec/model.py
qsdrec/train.py
```

新增参数：

```text
--hub-score-weight
--hub-attn-weight
--hub-loss-weight
```

含义：

```text
hub_score_weight:
  直接从 semantic score 中减去 candidate item 的 semantic hubness。

hub_attn_weight:
  在 query-token attention logits 中惩罚 hub semantic token。

hub_loss_weight:
  训练时惩罚 attention 对 hub token 的依赖。
```

Hubness 计算：

```text
H(token) = log(1 + token occurrence count)
H(item)  = sum H(token in item SID)
```

随后归一化到 0-1 区间。

### V2: Intent Evidence Gate

对应想法：方向 D，Intent Evidence Bottleneck。

新增参数：

```text
--evidence-gate none|history_overlap
--evidence-floor
```

当前实现：

```text
对于 candidate 的每个 SID token，
检查用户历史中是否存在同槽位相同 token。
如果存在，认为该 token 有历史证据支持；
如果不存在，则 attention 权重乘以 evidence_floor。
```

这相当于让模型显式区分：

```text
当前候选的哪些 semantic token 被用户历史支持；
哪些 token 只是全局热门或无证据语义。
```

### V3: Contrastive Semantic Decoding

对应想法：Contrastive Decoding。

新增参数：

```text
--contrastive-alpha
```

当前实现：

```text
expert semantic score:
  原 QSDRec semantic branch，包含 query-guided semantic matching。

amateur semantic score:
  简化语义匹配，用用户历史 semantic mean 与 candidate semantic mean 点积。

final semantic score:
  sem_score = expert_score - alpha * amateur_score
```

目标：

```text
压制 expert 和 amateur 都容易给高分的常见语义主题，
突出 query-guided branch 的差异化信号。
```

### V4: Combined Controls

组合版本只用于验证机制是否互补，不作为第一优先主结果。

建议先看单机制结果，再跑组合。

### V5: CR-SID Item-frequency Residual

对应想法：用一个统一 item representation 替代多分支 late fusion。

实现位置：

```text
qsdrec/model.py
qsdrec/train.py
scripts/train_crsid.py
```

新增参数：

```text
--model-variant crsid
--cr-tail-tau
--cr-residual-scale
--cr-residual-reg
```

当前实现：

```text
e_i = semantic_basis(z_i)
    + residual_scale * [alpha_i * private_item_residual
                        + (1 - alpha_i) * shared_semantic_residual]

alpha_i = item_train_frequency_i / (item_train_frequency_i + tau)
```

说明：

```text
该版本保留作为 CR-SID 的初始版本。
它关注 item-frequency long-tail，不直接针对 Semantic ID token 的头部 hub 问题。
```

### V6: CR-SID Semantic-hub Residual

对应想法：Semantic ID token 本身存在长尾/头部 hub，alpha 不再由 item 频次决定，而由 item 的 Semantic ID token hubness 决定。

实现位置：

```text
qsdrec/model.py
qsdrec/train.py
scripts/train_crsid_semhub.py
```

新增参数：

```text
--model-variant crsid_semhub
--cr-residual-scale
--cr-residual-reg
--cr-hub-alpha-floor
--cr-hub-alpha-gamma
```

当前实现：

```text
hub(z_i) = mean(normalized_log_frequency(z_i1), ..., normalized_log_frequency(z_i4))

alpha_i = hub_alpha_floor + (1 - hub_alpha_floor) * hub(z_i)^gamma

e_i = semantic_basis(z_i)
    + residual_scale * [alpha_i * private_item_residual
                        + (1 - alpha_i) * shared_semantic_residual]
```

直观解释：

```text
如果 item 的 Semantic ID token 多为热门 hub token，
说明 shared semantic residual 更容易混入泛办公主题和 shortcut，
因此提高 private residual 比例。

如果 item 的 Semantic ID token 更偏尾部，
说明共享语义更有区分度，
因此保留更多 shared semantic residual。
```

该版本更贴合当前 badcase 观察：

```text
问题不是普通 item long-tail，而是 Semantic ID token sharing imbalance。
```

## 2. 暂未实现或需单独分支的版本

### Global Semantic Anchor Tokenizer

对应想法：Semantic ID tokenizer 量化前引入全局语义锚点。

当前暂未直接实现，原因：

```text
现有 tokenizer 是离线 BGE embedding + RQ-KMeans，
没有训练目标，也没有可学习参数更新流程。
```

如果严格实现 learnable anchor，需要新增 tokenizer training 阶段，例如：

```text
item text embedding e_i
learnable anchor a
gamma_i = sigmoid(MLP(e_i))
e_i_aug = gamma_i * e_i + (1 - gamma_i) * a
then RQ-KMeans(e_i_aug)
```

需要额外设计训练目标，例如重构损失、review/series consistency 损失或 downstream proxy loss。

第一版可以做非学习近似版本：

```text
a = global centroid of item embeddings
gamma_i = quality score from metadata length / frequency / embedding isolation
```

但它不等价于你提出的 learnable anchor，需要单独命名。

### Shortcut Contrast / Tri-level Contrastive Set

对应想法：方向 B/C。

当前未直接实现，原因：

```text
需要构造 user-specific dominant semantic groups、
intent-supported positives、
ambiguous siblings、
shortcut negatives。
```

这会改动 sampler 和 loss，适合在 V1-V3 结果明确后单独实现。

### Hub Direction Removal / Representation Steering

对应想法：方向 E。

当前未直接实现，原因：

```text
需要定义用户历史 dominant semantic direction 和 candidate support direction。
```

它可以基于 V2 的 evidence gate 继续扩展。

### Semantic Hypergraph / Context-resolved Shared Code Node

对应想法：方向 F。

当前未直接实现，原因：

```text
需要重新组织 semantic code graph 或引入 code-context resolution module。
```

适合作为后续主方法版本，而不是和本轮控制实验混在一起。

## 3. 推荐运行顺序

Linux / 5090 服务器：

```bash
bash scripts/run_office_semantic_controls.sh
```

如果只想先跑最小集合：

```bash
EPOCHS=100 BATCH_SIZE=1024 bash scripts/run_office_semantic_controls.sh
```

优先观察：

```text
exp_hub_attn005_k8_sem010
exp_evidence_f020_k8_sem010
exp_contrastive005_k8_sem010
exp_evidence_hub_k8_sem010
exp_evidence_contrastive_k8_sem010
```

### 3.1 版本运行指令

以下命令都写入不同输出目录，不会覆盖已有版本。

#### V0: SASRec Baseline

```bash
/voice/bin/python scripts/train_qsdrec.py \
  --model-variant qsdrec \
  --dataset-dir runs/office \
  --semantic-ids runs/office/semantic_ids_rq.json \
  --output-dir runs/office/exp_sasrec \
  --device cuda \
  --epochs 100 \
  --early-stop-patience 10 \
  --batch-size 256 \
  --max-len 50 \
  --dim 128 \
  --num-interests 1 \
  --num-hard-neg 0 \
  --num-random-neg 100 \
  --sem-weight 0 \
  --dis-weight 0 \
  --div-weight 0
```

#### V0: QSDRec Reference

```bash
/voice/bin/python scripts/train_qsdrec.py \
  --model-variant qsdrec \
  --dataset-dir runs/office \
  --semantic-ids runs/office/semantic_ids_rq.json \
  --output-dir runs/office/exp_interest8_sem010 \
  --device cuda \
  --epochs 100 \
  --early-stop-patience 10 \
  --batch-size 256 \
  --max-len 50 \
  --dim 128 \
  --num-interests 8 \
  --num-hard-neg 0 \
  --num-random-neg 100 \
  --sem-weight 0.10 \
  --dis-weight 0 \
  --div-weight 0
```

#### V2: Binary Evidence

```bash
/voice/bin/python scripts/train_qsdrec.py \
  --model-variant qsdrec \
  --dataset-dir runs/office \
  --semantic-ids runs/office/semantic_ids_rq.json \
  --output-dir runs/office/exp_evi_binary_f020_k8_sem010 \
  --device cuda \
  --epochs 100 \
  --early-stop-patience 10 \
  --batch-size 256 \
  --max-len 50 \
  --dim 128 \
  --num-interests 8 \
  --num-hard-neg 0 \
  --num-random-neg 100 \
  --sem-weight 0.10 \
  --dis-weight 0 \
  --div-weight 0 \
  --evidence-gate history_overlap \
  --evidence-floor 0.20
```

#### V5: CR-SID Item-frequency Residual

```bash
/voice/bin/python scripts/train_crsid.py \
  --dataset-dir runs/office \
  --semantic-ids runs/office/semantic_ids_rq.json \
  --output-dir runs/office/crsid_itemfreq_tau20_s10 \
  --device cuda \
  --epochs 100 \
  --early-stop-patience 10 \
  --batch-size 256 \
  --max-len 50 \
  --dim 128 \
  --num-hard-neg 0 \
  --num-random-neg 100 \
  --cr-tail-tau 20 \
  --cr-residual-scale 1.0
```

#### V6: CR-SID Semantic-hub Residual

```bash
/voice/bin/python scripts/train_crsid_semhub.py \
  --dataset-dir runs/office \
  --semantic-ids runs/office/semantic_ids_rq.json \
  --output-dir runs/office/crsid_semhub_f005_g10_s10 \
  --device cuda \
  --epochs 100 \
  --early-stop-patience 10 \
  --batch-size 256 \
  --max-len 50 \
  --dim 128 \
  --num-hard-neg 0 \
  --num-random-neg 100 \
  --cr-hub-alpha-floor 0.05 \
  --cr-hub-alpha-gamma 1.0 \
  --cr-residual-scale 1.0
```

Semantic-hub 版本第一轮建议只改：

```text
crsid_semhub_f005_g10_s10
crsid_semhub_f010_g10_s10
crsid_semhub_f005_g05_s10
crsid_semhub_f005_g20_s10
crsid_semhub_f005_g10_s05
crsid_semhub_f005_g10_s20
```

## 4. 结果分析流程

汇总全局结果：

```powershell
python scripts\summarize_experiments.py `
  --root runs\office `
  --metric NDCG@10 `
  --top-k 30 `
  --csv runs\office\semantic_control_summary.csv
```

对最优模型做 slot-overlap 分组：

```powershell
python scripts\compare_semantic_sharing_groups.py `
  --base-checkpoint runs\office\exp_sasrec\best.pt `
  --compare-checkpoint runs\office\exp_evidence_hub_k8_sem010\best.pt `
  --output runs\office\semantic_overlap2_group_compare_evidence_hub.json `
  --dataset-dir runs\office `
  --semantic-ids runs\office\semantic_ids_rq.json `
  --sharing-mode overlap `
  --min-overlap-slots 2 `
  --buckets "1,2-5,6-10,11-20,21-50,>50"
```

挖 badcase：

```powershell
python scripts\mine_bad_cases.py `
  --base-checkpoint runs\office\exp_interest8_sem010\best.pt `
  --compare-checkpoint runs\office\exp_evidence_hub_k8_sem010\best.pt `
  --output runs\office\bad_cases_nohard_vs_evidence_hub_overlap2.json `
  --dataset-dir runs\office `
  --semantic-ids runs\office\semantic_ids_rq.json `
  --sharing-mode overlap `
  --min-overlap-slots 2 `
  --min-overlap-size 2 `
  --top-k 10 `
  --num-cases 80
```

生成 Markdown 摘要：

```powershell
python scripts\summarize_bad_cases.py `
  --input runs\office\bad_cases_nohard_vs_evidence_hub_overlap2.json `
  --output runs\office\bad_cases_nohard_vs_evidence_hub_overlap2.md
```

## 5. 论文解释主线

本轮机制服务于同一个问题定义：

```text
Token-level Semantic Sharing Imbalance
```

包含：

```text
Under-sharing:
  真实相关 item 没有共享足够 semantic code。

Over-sharing:
  hub semantic token 或热门语义主题过度影响 semantic branch。
```

V1/V2/V3 分别对应：

```text
V1: 抑制 hub token / hub item。
V2: 只强化有用户历史证据支持的 candidate token。
V3: 减去 amateur semantic matcher 捕获到的常见语义偏置。
```
