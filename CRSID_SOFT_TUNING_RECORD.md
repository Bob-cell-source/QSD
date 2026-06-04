# CRSID Soft SID 调参记录

## 1. 实验背景

本轮实验目标是验证在 hard CRSID 基础上，将 hard Semantic ID 改造成 local-consistent soft Semantic ID 是否能进一步提升效果。

原 hard CRSID 使用每个 item 的单一路径 Semantic ID：

$$
z_i = [z_{i,1}, z_{i,2}, z_{i,3}, z_{i,4}]
$$

soft SID 版本为每个 item/slot 构造多个候选 token，并根据局部邻居中的 token 支持度进行加权：

$$
Z_{i,l} = \{(c_{i,l,m}, \tilde{p}(c_{i,l,m} \mid i,l))\}_{m=1}^{M}
$$

然后用 soft SID 替代 hard SID pooling，构造 semantic basis 和 shared semantic residual。

本轮结果来自 Beauty 数据集：

```text
runs/beauty/crsid_soft_probe/summary.csv
runs/beauty/crsid_soft_tuning/summary.csv
```

核心训练配置：

```text
dataset = Beauty
dim = 128
batch_size = 1024
num_random_neg = 100
num_hard_neg = 0
max_len = 50
lr = 1e-3
weight_decay = 1e-4
seed = 2026
```

## 2. 已有主结果

| Experiment | NDCG@10 | HR@10 | NDCG@20 | 说明 |
|---|---:|---:|---:|---|
| `00_sasrec_id_only` | 0.044830 | 0.077047 | 0.052492 | 纯 ID SASRec baseline |
| `01_qsdrec_semantic_score` | 0.042957 | 0.074453 | 0.050142 | QSD score-level semantic fusion |
| `10_crsid_hard_tau20_s10` | 0.051643 | 0.089925 | 0.061100 | hard SID CRSID |
| `21_crsid_soft_m4_s005_rel010` | 0.052266 | 0.091625 | 0.061346 | 原 soft SID 最优 |
| `27_crsid_soft_m4_s005_prior1_eta2_n50` | 0.052575 | 0.091177 | 0.061586 | 当前 soft SID 最优 |

当前最优为：

```text
27_crsid_soft_m4_s005_prior1_eta2_n50
```

相对 hard CRSID：

```text
NDCG@10: 0.052575 vs 0.051643, +0.000932, +1.80%
HR@10:   0.091177 vs 0.089925, +0.001252, +1.39%
NDCG@20: 0.061586 vs 0.061100, +0.000486, +0.80%
```

相对 SASRec：

```text
NDCG@10: 0.052575 vs 0.044830, +17.28%
```

相对 QSD semantic-score baseline：

```text
NDCG@10: 0.052575 vs 0.042957, +19.64%
```

## 3. Soft SID 调参结果

| Experiment | NDCG@10 | HR@10 | NDCG@20 | 结论 |
|---|---:|---:|---:|---|
| `21_crsid_soft_m4_s005_rel010` | 0.052266 | 0.091625 | 0.061346 | soft SID 主候选，优于 hard CRSID |
| `24_crsid_soft_m4_s005_rel1` | 0.052251 | 0.092385 | 0.061332 | 关闭 reliability alpha 后几乎不变 |
| `22_crsid_soft_m4_s010_rel010` | 0.051861 | 0.090775 | 0.060894 | support 阈值 0.10 偏严格，下降 |
| `20_crsid_soft_m4_no_prune_rel1` | 0.051834 | 0.090417 | 0.060914 | 不做局部剪枝也能略高于 hard，但不如 local consistency |
| `23_crsid_soft_m8_s005_rel010` | 0.051410 | 0.089120 | 0.060569 | top-M 太大，引入 noisy sharing |
| `27_crsid_soft_m4_s005_prior1_eta2_n50` | 0.052575 | 0.091177 | 0.061586 | 当前最优，support eta=2 有效 |
| `26_crsid_soft_m4_s005_prior2_eta1_n50` | 0.051957 | 0.088807 | 0.061178 | hard token prior=2 过度相信 hard SID |
| `25_crsid_soft_m3_s005_prior1_eta1_n50` | 0.051372 | 0.088494 | 0.060171 | top-M=3 候选太少 |
| `28_crsid_soft_m4_s005_prior1_eta1_n20` | 0.051258 | 0.088181 | 0.060150 | neighbor=20 太少，局部支持估计不稳 |

## 4. 关键结论

### 4.1 Soft SID 方向有效，但提升是小幅稳定提升

hard CRSID 已经明显优于 SASRec 和 QSD。Local-consistent soft SID 在 hard CRSID 上进一步提升：

```text
hard CRSID NDCG@10 = 0.051643
best soft CRSID NDCG@10 = 0.052575
```

这说明 hard Semantic ID 的确存在 assignment 过硬的问题，保留有限候选 token 可以缓解 under-sharing。

### 4.2 Local consistency 是必要的

对比：

```text
20 no prune:     NDCG@10 = 0.051834
21 local s=0.05: NDCG@10 = 0.052266
27 eta=2:        NDCG@10 = 0.052575
```

说明 soft candidate SID 不能无约束使用。需要用局部邻居支持度筛选或加权候选 token，否则容易引入 over-sharing 噪声。

### 4.3 top-M 不是越大越好

```text
m=4: NDCG@10 = 0.052575
m=8: NDCG@10 = 0.051410
```

候选 token 过多会重新引入假共享和热门 token 噪声。当前 Beauty 上 `top_m=4` 更合适。

### 4.4 过度相信 hard token 会削弱 soft SID 的修正能力

```text
prior=1: NDCG@10 = 0.052575
prior=2: NDCG@10 = 0.051957
```

这说明 hard SID 确实存在错配。soft candidate 的作用是修正 hard assignment，不能把 hard token 权重压得过强。

### 4.5 reliability alpha 当前不是核心收益来源

```text
21 rel_floor=0.10: NDCG@10 = 0.052266
24 rel_floor=1.00: NDCG@10 = 0.052251
```

两者几乎一致。当前主要收益来自 local-consistent soft SID，而不是 reliability-calibrated alpha。论文中不建议把 reliability alpha 写成核心贡献。

## 5. 当前推荐主方法配置

建议当前主方法采用：

```text
model_variant = crsid_soft
cr_tail_tau = 20
cr_residual_scale = 1.0
cr_soft_top_m = 4
cr_soft_min_overlap_slots = 2
cr_soft_min_support = 0.05
cr_soft_support_eta = 2.0
cr_soft_hard_token_prior = 1.0
cr_soft_reliability_floor = 0.10
cr_soft_max_neighbors = 50
```

实验名：

```text
27_crsid_soft_m4_s005_prior1_eta2_n50
```

方法命名建议：

```text
LC-Soft CRSID
Local-Consistent Soft Collaborative-Residual Semantic ID Representation
```

## 6. 论文叙事建议

推荐将论文主线从“frequency-adaptive CRSID”调整为：

```text
Hard Semantic ID suffers from under-sharing and over-sharing.
We introduce local-consistent soft Semantic IDs to preserve limited assignment uncertainty
while suppressing locally unsupported token sharing.
The resulting soft SID is then used in a collaborative-residual item representation.
```

中文表述：

```text
现有 hard Semantic ID 将每个 item 强制分配到单一路径，这会带来两类问题：
一方面，真实同系列 item 可能因为量化误差被分到不同 token，导致 under-sharing；
另一方面，热门 token 覆盖大量语义区域，容易造成 over-sharing 和语义漂移。
为此，本文提出 local-consistent soft SID，为每个 item/slot 保留有限候选 token，
并利用局部邻居中的 token 支持度对候选进行重加权，从而在保留语义不确定性的同时抑制假共享。
```

## 7. 后续建议

当前结果已经可以支撑方法方向，但提升幅度仍然属于 modest improvement。建议下一步：

```text
1. 在 Office 上补跑同样 soft SID 配置，验证是否也优于 hard CRSID。
2. 在 Beauty 上跑多 seed，至少 seed=2026, 2027, 2028。
3. 最终论文主表用 LC-Soft CRSID，消融表保留 hard CRSID / no prune / m=8 / eta=1 / eta=2。
4. reliability alpha 不作为主贡献，只作为实现中的辅助项或放入附录。
```
