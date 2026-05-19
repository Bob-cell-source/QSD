# QSDRec 项目说明与阶段性实验总结

本文档用于记录当前论文项目的代码框架、模型实现、实验设置、对比分析结果和阶段性结论。当前项目重点研究 Semantic ID 在序列推荐中的作用机制，尤其关注低频物品、共享语义码、语义漂移和用户侧语义兴趣建模问题。

## 1. 项目目标

本项目以序列推荐为任务场景，基础模型为 SASRec，在此基础上引入 item 侧 Semantic ID 和用户侧语义兴趣查询模块，形成 QSDRec 框架。

当前研究问题可以概括为：

```text
Semantic ID 能通过 token-level sharing 帮助低频 item 泛化，
但 naive semantic branch 也会受到不可靠语义共享和热门语义主题影响，
导致 semantic drift。
```

因此，本项目当前重点不是简单验证“语义分支是否有效”，而是进一步分析：

- Semantic ID 的层级和 token 共享到底学到了什么。
- QSDRec 相比纯 ID 方法在哪些场景有效。
- QSDRec 在哪些 badcase 中失败，失败是否来自 under-sharing 或 over-sharing。
- prefix sharing 和任意槽位 slot-overlap sharing 是否都能解释模型行为。
- hard negative 是否能缓解语义混淆，还是会压制语义泛化。

## 2. 代码框架

主要目录如下：

```text
qsdrec/
  preprocess.py        Amazon 数据预处理，生成 sequences/item_meta/stats
  semantic_id.py       文本编码与 RQ-KMeans Semantic ID 构建、分析
  model.py             SASRec encoder 与 QSDRec 模型
  train.py             训练、负采样、全量验证/测试
  io_utils.py          JSON/JSONL 工具

scripts/
  train_qsdrec.py                         训练入口
  build_semantic_ids.py                   Semantic ID 构建入口
  summarize_experiments.py                汇总 test_metrics
  plot_history.py                         绘制 loss/valid 曲线
  evaluate_high_sharing_groups.py         prefix-sharing 分组评估
  evaluate_cold_start_groups.py           低频/冷启动分组评估
  compare_semantic_sharing_groups.py      prefix/slot-overlap 分组对比
  analyze_semantic_levels.py              Semantic ID 层级解释
  analyze_overlap_samples.py              slot-overlap 样本解释
  mine_bad_cases.py                       badcase 挖掘
  plot_semantic_clusters.py               Semantic ID 聚类可视化
```

## 3. 数据与预处理

目前主要使用 Amazon Office 和 Beauty 数据。

Office 数据统计：

```text
num_users: 4905
num_items: 2420
num_interactions: 53258
min_user_inter: 5
min_item_inter: 5
```

Beauty 数据统计：

```text
num_users: 22363
num_items: 12101
num_interactions: 198502
min_user_inter: 5
min_item_inter: 5
```

预处理后每个数据集包含：

```text
sequences.json      用户按时间排序后的 item 序列
item_meta.json      item 的 asin/title/brand/categories/description
stats.json          数据统计
item2id.json        item 映射
user2id.json        user 映射
```

## 4. Semantic ID 构建

当前 Semantic ID 构建方式为：

```text
item metadata text
  -> BGE text encoder
  -> item text embedding
  -> RQ-KMeans residual quantization
  -> 4-level semantic ID
```

默认码本大小：

```text
64,128,256,512
```

注意：当前实现不是人工定义 taxonomy，也不是端到端 tokenizer，而是离线文本 embedding 上的 residual quantization。第 1 层聚类原始 embedding，第 2/3/4 层聚类上一层无法解释的 residual。

Office Semantic ID 层级统计：

```text
Level 1: unique=64,   avg_size=37.81, max_size=106, collision_rate=0.9736
Level 2: unique=437,  avg_size=5.54,  max_size=26,  collision_rate=0.8194
Level 3: unique=1662, avg_size=1.46,  max_size=8,   collision_rate=0.3132
Level 4: unique=2151, avg_size=1.13,  max_size=4,   collision_rate=0.1112
```

Beauty Semantic ID 层级统计：

```text
Level 1: unique=64,   avg_size=189.08, max_size=533, collision_rate=0.9947
Level 2: unique=1217, avg_size=9.94,   max_size=68,  collision_rate=0.8994
Level 3: unique=4719, avg_size=2.56,   max_size=27,  collision_rate=0.6100
Level 4: unique=9282, avg_size=1.30,   max_size=15,  collision_rate=0.2330
```

层级解释实验显示：

```text
Level 1: 粗粒度语义大类，例如 printer ink、tape、hair care、fragrance
Level 2: 子品类 / 品牌 / 产品线，例如 HP ink、Pilot G2、shampoo
Level 3: 更细的系列 / 功能 / 规格
Level 4: 接近 SKU 级或近重复物品
```

## 5. 模型框架

QSDRec 当前由两部分组成：

```text
ID branch:
  SASRec encoder 生成用户 ID 行为表示 h_id
  h_id 与 candidate item embedding 点积得到 id_score

Semantic branch:
  item semantic ID 被映射为 semantic token embeddings
  用户历史 item 的 semantic token 聚合为 semantic memory
  query prototypes 从 semantic memory 中提取多个用户语义兴趣
  query 与 candidate semantic tokens 做 attention
  得到 sem_score

Final score:
  score(u, i) = id_score(u, i) + sem_weight * sem_score(u, i)
```

当前语义分支并不是只学习前缀共享。candidate 的完整 semantic ID `[c1,c2,c3,c4]` 都会参与语义打分，因此理论上能够学习任意槽位上的 semantic code 相似性。

但当前模型没有显式区分以下几类共享：

```text
prefix sharing:       共享连续前缀，例如 same [c1,c2]
slot-overlap sharing: 任意相同槽位数量 >= 2
full-SID sharing:     完整 semantic ID 相同
```

这也是后续改进的重要切入点。

## 6. 训练与评估设置

当前训练默认使用 sampled objective：

```text
1 positive + num_hard_neg + num_random_neg
```

验证和测试均使用全量 item ranking：

```text
valid: full-ranking NDCG@10 选择 best.pt
test:  加载 best.pt 后 full-ranking 评估
```

评估指标：

```text
HR@5/10/20
Recall@5/10/20
NDCG@5/10/20
```

已经修复过一个重要 bug：历史序列 padding 位置不再错误映射到 item ID 1，避免 item 1 被永久 mask。

## 7. 主要对比实验

### 7.1 SASRec 与 QSDRec

Office 上当前代表性结果：

```text
SASRec:
NDCG@10 = 0.057078
HR@10   = 0.103568

QSDRec, sem_weight=0.1, num_interests=8, no hard neg:
NDCG@10 = 0.063522
HR@10   = 0.112130
```

结论：

```text
语义分支整体有效，相比纯 ID branch 有明显提升。
```

### 7.2 Semantic Weight 消融

Office 部分结果：

```text
sem_weight=0.05: NDCG@10 = 0.057360
sem_weight=0.10: NDCG@10 = 0.061751
sem_weight=0.20: NDCG@10 = 0.058318
sem_weight=0.50: NDCG@10 = 0.061595
sem_weight=1.00: NDCG@10 = 0.062452
```

结论：

```text
语义分支不是越强越好，需要控制语义贡献。
```

### 7.3 Multi-interest 消融

Office batch=256 结果中，`num_interests=8` 最优：

```text
K=1: NDCG@10 = 0.059922
K=2: NDCG@10 = 0.059347
K=4: NDCG@10 = 0.061751
K=8: NDCG@10 = 0.063522
```

结论：

```text
多个语义兴趣 query 有助于捕获用户多样偏好，但并不自动解决语义漂移。
```

### 7.4 Full-softmax 训练

曾尝试使用全量 softmax 训练，但代表性结果没有超过 sampled 训练：

```text
exp_interest1_sem010_fullsoftmax:
NDCG@10 = 0.061034
```

当前训练策略已回到 sampled objective。

### 7.5 Hard Negative 消融

当前新增两种 hard negative：

```text
prefix hard negative:
  与 target 共享前 prefix_level 个 code

overlap hard negative:
  与 target 在相同槽位上至少 min_overlap_slots 个 code 相同
```

Office 上结果：

```text
no hard neg:
NDCG@10=0.063522, HR@10=0.112130

prefix hard5:
NDCG@10=0.059821, HR@10=0.107034

overlap hard5:
NDCG@10=0.062272, HR@10=0.110703

overlap hard10:
NDCG@10=0.062124, HR@10=0.109480
```

结论：

```text
overlap hard negative 明显优于 prefix hard negative，
但当前简单 hard negative 仍未超过无 hard negative。
hard negative 过强可能压制语义泛化。
```

## 8. 分组实验

### 8.1 High Prefix-sharing Group

定义：

```text
对于测试样本的 ground-truth target item，
统计与 target 共享 semantic ID 前两个 code 的 item 数量。
```

Office 结果：

```text
prefix_size=1:
SASRec 0.051156 -> QSD 0.042617  -0.008539

prefix_size=2-5:
SASRec 0.055680 -> QSD 0.052929  -0.002751

prefix_size=6-10:
SASRec 0.055427 -> QSD 0.066921  +0.011494

prefix_size>10:
SASRec 0.059819 -> QSD 0.067777  +0.007958
```

结论：

```text
QSDRec 在共享前缀邻居充足时有效；
当 target 是 prefix 单例或小共享组时，语义分支可能有害。
```

### 8.2 Slot-overlap Sharing Group

定义：

```text
如果两个 item 在相同槽位上至少有 2 个 semantic code 相同，
则认为存在 slot-overlap sharing。
```

Office 结果：

```text
overlap_size=2-5:
SASRec 0.090535 -> QSD 0.078679  -0.011856

overlap_size=6-10:
SASRec 0.028081 -> QSD 0.045796  +0.017715

overlap_size=11-20:
SASRec 0.058915 -> QSD 0.066648  +0.007733

overlap_size=21-50:
SASRec 0.062766 -> QSD 0.067650  +0.004884

overlap_size>50:
SASRec 0.046970 -> QSD 0.053241  +0.006272
```

结论：

```text
slot-overlap sharing 也能解释 QSD 的收益和失败。
这说明问题不应局限为 shared-prefix ambiguity，
而应扩展为 semantic-code sharing reliability。
```

### 8.3 Low-frequency / Cold-start Group

定义：

```text
统计 target item 在训练集中作为 next-item target 出现的次数。
```

Office 结果：

```text
train_count=1-5:
SASRec 0.013857 -> QSD 0.022050  +59.13%

train_count=6-10:
SASRec 0.024986 -> QSD 0.029806  +19.29%

train_count=11-20:
SASRec 0.058961 -> QSD 0.071622  +21.47%

train_count>20:
SASRec 0.104358 -> QSD 0.107723  +3.22%
```

结论：

```text
语义分支对低频和长尾 target 有明显帮助，
但当低频 target 在 semantic ID 空间中也是孤立点时，仍然容易失败。
```

## 9. Badcase 分析

已生成的 badcase 文件包括：

```text
runs/office/bad_cases_sasrec_vs_qsd_highsharing_more.json
runs/office/bad_cases_sasrec_vs_qsd_lowfreq_more.json
runs/office/bad_cases_sasrec_vs_qsd_overlap2_more.json
runs/office/bad_cases_sasrec_vs_qsd_overlaphard5_overlap2.json
runs/office/bad_cases_qsd_nohard_vs_overlaphard5_overlap2.json
```

badcase 统计方式：

```text
对每个 test sample 做 full-ranking Top@10。
比较 base model 和 compare model 是否命中 target。
分为：
  base_correct_comp_wrong
  comp_correct_base_wrong
  both_correct
  both_wrong
```

每条样本保存：

```text
target item metadata
history tail
base top items
compare top items
semantic_id
prefix_group_size
overlap_group_size
train_count
```

### 9.1 QSD 成功模式

典型成功案例：

```text
Epson WorkForce printer series
HP / Canon ink cartridge series
Quartet dry erase board / marker
Wilson Jones binder
Rolodex mesh organizer
```

原因：

```text
低频 target 的 ID embedding 不容易被纯 ID 方法记住，
但同系列或同语义邻居共享 semantic code，
这些邻居在训练中为 semantic token 提供梯度，
因此 QSD 能把 target 拉进 Top10。
```

### 9.2 QSD 失败模式

典型失败案例：

```text
HP 940XL Cyan Ink Cartridge
Five Star Locker Light
Fellowes Laminator Neptune3
Mead Notebook
Canon Photo Paper
```

主要原因：

```text
Under-sharing:
真实同系列 item 没有被分到同一个 prefix group。
例如 HP 940XL Cyan 是 [1,52,140,388]，
历史中 HP 940XL Yellow/Magenta 是 [1,43,140,...]。
如果只看 prefix，target 是孤立点。

Over-sharing:
用户历史中的热门语义主题更强，
例如 marker、label、binder、folder 等办公用品主题，
导致 semantic branch 将候选分数推向这些更密集的语义区域。
```

因此当前观察到两个相反问题：

```text
共享不足：真实相关 item 没有足够 semantic-code sharing。
共享过多：热门语义 token 或大语义簇支配用户语义兴趣。
```

## 10. 阶段性结论

当前项目已经形成以下核心结论：

1. Semantic ID 分支整体有效，尤其对低频 item 和有可靠语义邻居的 target 有帮助。

2. Semantic ID 的有效性来自 token-level sharing。多个 item 共享 semantic code 时，会共同更新这些 token embedding，从而帮助低频 target 泛化。

3. Prefix sharing 不是唯一有效共享形式。对于 RQ-KMeans residual semantic ID，slot-overlap sharing 同样能解释模型收益。

4. 当前 QSDRec 同时存在 under-sharing 和 over-sharing 问题。前者导致低频 target 无法借力，后者导致 semantic branch 被历史中的热门语义主题带偏。

5. 简单 hard negative 不能稳定解决问题。Overlap hard negative 优于 prefix hard negative，但仍未超过无 hard negative，说明问题不只是“负样本不够难”，而是需要判断 semantic sharing 是否可靠。

6. 后续方法应从简单增强语义分支，转向 reliability-aware semantic sharing。

## 11. 后续改进方向

当前最合理的论文方向是：

```text
Reliability-aware Semantic-code Sharing for Sequential Recommendation
```

或：

```text
Token-level Semantic Sharing Imbalance in Semantic-ID based Recommendation
```

建议从两条线推进：

### 11.1 编码侧改进

目标：

```text
缓解 under-sharing，使真实同系列 / 同功能 item 获得可靠语义邻居。
```

可考虑：

```text
prefix-sharing edge
slot-overlap sharing edge
brand/title-series evidence
category evidence
semantic-code reliability score
```

重点不是简单重新聚类，而是校准 semantic-code sharing 的可靠性。

### 11.2 用户侧查询改进

目标：

```text
缓解 over-sharing 和 semantic drift。
```

建议：

```text
candidate-specific evidence retrieval
semantic reliability gate
history evidence count
prefix_group_size / overlap_group_size aware gating
```

形式上可以写为：

```text
score(u, i) = id_score(u, i) + g(u, i) * sem_score_evidence(u, i)
```

其中 `g(u,i)` 用于判断当前 candidate 的语义分支是否可靠。

## 12. 常用命令

汇总实验：

```powershell
python scripts\summarize_experiments.py `
  --root runs\office `
  --metric NDCG@10 `
  --top-k 20 `
  --csv runs\office\experiment_summary.csv
```

分析 Semantic ID 层级：

```powershell
python scripts\analyze_semantic_levels.py `
  --semantic-ids runs\office\semantic_ids_rq.json `
  --item-meta runs\office\item_meta.json `
  --output runs\office\semantic_level_explanation.json
```

比较 slot-overlap 分组：

```powershell
python scripts\compare_semantic_sharing_groups.py `
  --base-checkpoint runs\office\exp_sasrec\best.pt `
  --compare-checkpoint runs\office\exp_interest8_sem010\best.pt `
  --output runs\office\semantic_overlap2_group_compare.json `
  --dataset-dir runs\office `
  --semantic-ids runs\office\semantic_ids_rq.json `
  --sharing-mode overlap `
  --min-overlap-slots 2 `
  --buckets "1,2-5,6-10,11-20,21-50,>50"
```

挖 overlap badcase：

```powershell
python scripts\mine_bad_cases.py `
  --base-checkpoint runs\office\exp_sasrec\best.pt `
  --compare-checkpoint runs\office\exp_interest8_sem010\best.pt `
  --output runs\office\bad_cases_sasrec_vs_qsd_overlap2_more.json `
  --dataset-dir runs\office `
  --semantic-ids runs\office\semantic_ids_rq.json `
  --sharing-mode overlap `
  --min-overlap-slots 2 `
  --min-overlap-size 10 `
  --top-k 10 `
  --num-cases 80
```
