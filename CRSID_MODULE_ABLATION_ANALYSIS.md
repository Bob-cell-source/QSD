# CRSID 模块消融实验分析

实验结果来自：

```text
runs/office/crsid_module_ablation
```

汇总文件：

```text
runs/office/crsid_module_ablation/summary.csv
```

本轮实验使用 Office 数据集，核心配置为：

```text
dim = 128
batch_size = 256
num_random_neg = 100
num_hard_neg = 0
epochs = 30
early_stop_patience = 5
seed = 2026
metric = full-ranking NDCG@10 / HR@10 / NDCG@20
```

## 1. 总体结果

按测试集 NDCG@10 排序：

| Rank | Experiment | NDCG@10 | HR@10 | NDCG@20 | Best Valid NDCG@10 |
|---:|---|---:|---:|---:|---:|
| 1 | `10_crsid_full_tau20_s10` | 0.067048 | 0.117023 | 0.081066 | 0.082519 |
| 2 | `12_crsid_no_semantic_basis` | 0.064856 | 0.108869 | 0.078128 | 0.077641 |
| 3 | `13_crsid_no_shared_residual` | 0.064700 | 0.111519 | 0.078871 | 0.080821 |
| 4 | `15_crsid_fixed_alpha_050` | 0.061736 | 0.111315 | 0.076142 | 0.079822 |
| 5 | `14_crsid_no_private_residual` | 0.061424 | 0.111111 | 0.073420 | 0.072748 |
| 6 | `16_crsid_private_only_alpha_100` | 0.061063 | 0.104791 | 0.074993 | 0.078595 |
| 7 | `20_crsid_semhub_full_f005_g10_s10` | 0.060424 | 0.108053 | 0.074499 | 0.078656 |
| 8 | `01_qsdrec_semantic_score` | 0.060348 | 0.108461 | 0.075260 | 0.074787 |
| 9 | `00_sasrec_id_only` | 0.058792 | 0.104383 | 0.072496 | 0.074699 |
| 10 | `17_crsid_shared_only_alpha_000` | 0.055874 | 0.104587 | 0.067061 | 0.063200 |
| 11 | `11_crsid_basis_only_no_residual` | 0.051820 | 0.099490 | 0.063886 | 0.061683 |

完整 CRSID 在本轮模块消融中排名第一。

相对 baseline：

```text
CRSID full vs SASRec ID-only:
  NDCG@10: 0.067048 vs 0.058792
  relative gain: +14.04%

CRSID full vs QSDRec semantic-score baseline:
  NDCG@10: 0.067048 vs 0.060348
  relative gain: +11.10%
```

这说明 CRSID 的提升不是简单来自更强的训练配置，也不是简单增加 semantic score 分支，而是来自将 Semantic ID 内化为 item representation 的表示构造方式。

## 2. 完整 CRSID 与各消融版本的差值

以 `10_crsid_full_tau20_s10` 为参照：

| Experiment | NDCG@10 | Delta | Relative Drop | HR@10 Delta | Interpretation |
|---|---:|---:|---:|---:|---|
| `12_crsid_no_semantic_basis` | 0.064856 | -0.002192 | -3.27% | -0.008155 | 去掉 semantic basis 后下降 |
| `13_crsid_no_shared_residual` | 0.064700 | -0.002348 | -3.50% | -0.005505 | 去掉 shared residual 后下降 |
| `15_crsid_fixed_alpha_050` | 0.061736 | -0.005312 | -7.92% | -0.005708 | 固定 alpha 后明显下降 |
| `14_crsid_no_private_residual` | 0.061424 | -0.005624 | -8.39% | -0.005912 | 去掉 private residual 后明显下降 |
| `16_crsid_private_only_alpha_100` | 0.061063 | -0.005985 | -8.93% | -0.012232 | 只用 private residual 明显不足 |
| `20_crsid_semhub_full_f005_g10_s10` | 0.060424 | -0.006624 | -9.88% | -0.008970 | semantic-hub alpha 当前不如 item-frequency alpha |
| `01_qsdrec_semantic_score` | 0.060348 | -0.006699 | -9.99% | -0.008563 | QSDRec semantic score baseline 不如 CRSID |
| `00_sasrec_id_only` | 0.058792 | -0.008256 | -12.31% | -0.012640 | 纯 ID 序列模型不如 CRSID |
| `17_crsid_shared_only_alpha_000` | 0.055874 | -0.011174 | -16.67% | -0.012436 | 只用 shared residual 明显不足 |
| `11_crsid_basis_only_no_residual` | 0.051820 | -0.015227 | -22.71% | -0.017533 | 只用 semantic basis 最弱 |

## 3. 模块有效性分析

### 3.1 整体方法有效

完整 CRSID 明显优于两个基础对照：

```text
SASRec ID-only:              NDCG@10 = 0.058792
QSDRec semantic-score:       NDCG@10 = 0.060348
CRSID full:                  NDCG@10 = 0.067048
```

这支持主结论：

```text
Compared with using Semantic IDs as an auxiliary semantic scoring branch,
internalizing Semantic IDs into item representation yields better recommendation performance.
```

中文可写为：

```text
相比纯 ID 序列推荐和外加语义打分分支的 QSDRec baseline，完整 CRSID 均取得更高的 NDCG@10，
说明将 Semantic ID 直接用于 item 表示构造比将其作为额外 semantic score 更有效。
```

### 3.2 Collaborative residual 是关键模块

`11_crsid_basis_only_no_residual` 只保留 semantic basis，关闭整个 residual：

```text
CRSID full:       0.067048
Basis only:       0.051820
Drop:            -22.71%
```

这是最大幅度下降。说明只有 Semantic ID basis 不足以完成推荐任务，协同过滤残差是 CRSID 的核心收益来源。

可写为：

```text
The semantic basis alone performs poorly, indicating that Semantic IDs cannot replace collaborative signals.
The collaborative residual is necessary for adapting semantic representations to user-item interaction patterns.
```

### 3.3 Semantic basis 有效，但不是唯一主力

`12_crsid_no_semantic_basis` 去掉 semantic basis，只保留 residual path：

```text
CRSID full:             0.067048
No semantic basis:      0.064856
Drop:                  -3.27%
```

下降幅度不如去掉 residual 大，但仍然稳定下降。说明 semantic basis 对最终 item representation 有贡献，但 residual path 本身已经很强。

实验含义：

```text
semantic basis 提供 item 的语义基础位置；
residual path 提供协同适配；
二者组合优于只用 residual path。
```

论文中建议不要把 semantic basis 夸大成唯一核心，而应表述为“稳定的语义锚点”：

```text
Removing the semantic basis degrades performance, showing that the semantic basis provides a useful anchor for item representation.
However, the degradation is smaller than removing the residual, suggesting that semantic basis and collaborative residual play complementary roles.
```

### 3.4 Shared semantic residual 有效

`13_crsid_no_shared_residual` 去掉 Semantic ID token 共享残差：

```text
CRSID full:              0.067048
No shared residual:      0.064700
Drop:                   -3.50%
```

这说明 shared semantic residual 是有效模块。它让共享 Semantic ID token 的 item 可以共享一部分协同残差信号，尤其对交互较少的 item 有理论意义。

但需要注意：去掉 shared residual 后仍然排第 3，说明 private residual + semantic basis 仍然较强。shared residual 的贡献是增益项，不是唯一决定项。

可写为：

```text
Removing the shared Semantic-ID residual reduces NDCG@10 by 3.50%, confirming that token-level residual sharing contributes useful collaborative transfer.
```

### 3.5 Private item residual 更关键

`14_crsid_no_private_residual` 去掉 item 私有残差：

```text
CRSID full:              0.067048
No private residual:     0.061424
Drop:                   -8.39%
```

下降幅度大于去掉 shared residual。这说明在 Office 数据集上，item-specific collaborative signal 仍然非常重要。Semantic ID 共享不能完全替代 item 私有协同信息。

这个结果适合支持 CRSID 的“残差分解”设计：

```text
private residual 保留头部 item 或行为模式清晰 item 的个性化协同偏移；
shared residual 为长尾或语义相近 item 提供共享迁移；
完整模型需要二者同时存在。
```

### 3.6 Adaptive alpha 是强证据

固定 alpha 的结果：

```text
CRSID full:              0.067048
Fixed alpha = 0.5:       0.061736
Drop:                   -7.92%
```

只使用某一路径的结果：

```text
Private only alpha=1.0:  0.061063
Shared only alpha=0.0:   0.055874
```

完整 CRSID 同时优于固定混合、纯 private、纯 shared。这是本轮消融中最有价值的证据之一，说明 adaptive alpha 不是形式设计，而是确实带来了性能提升。

可写为：

```text
A fixed private/shared mixture is substantially worse than the adaptive mixture.
Moreover, both all-private and all-shared variants underperform the full model,
demonstrating that CRSID benefits from item-specific residual allocation.
```

中文可写为：

```text
固定 alpha 和单一路径 residual 均显著低于完整 CRSID，说明模型需要根据 item 状态自适应地分配 private residual 与 shared residual 的比例。
```

### 3.7 当前数据上 item-frequency alpha 优于 semantic-hubness alpha

两种完整 CRSID：

```text
Item-frequency alpha CRSID:    0.067048
Semantic-hubness alpha CRSID:  0.060424
Drop:                         -9.88%
```

这说明在当前 Office 实验设置下，使用 item 训练频次控制 alpha 更稳。`crsid_semhub` 可以作为诊断版本保留，但不建议作为当前主方法。

谨慎解释：

```text
The semantic-hubness alpha variant underperforms the item-frequency alpha variant on Office,
suggesting that item-level interaction frequency is a more reliable signal for residual allocation in this setting.
```

不要写成“semantic hubness 无效”。更准确的说法是：

```text
semantic-hubness alpha 在当前参数和数据集上不如 item-frequency alpha；
但它仍然提供了分析 Semantic ID token hubness 问题的诊断视角。
```

## 4. Early Stopping 与稳定性观察

各实验 best epoch：

| Experiment | Epochs Run | Best Epoch | Best Valid NDCG@10 | Test NDCG@10 |
|---|---:|---:|---:|---:|
| `00_sasrec_id_only` | 15 | 10 | 0.074699 | 0.058792 |
| `01_qsdrec_semantic_score` | 11 | 6 | 0.074787 | 0.060348 |
| `10_crsid_full_tau20_s10` | 25 | 20 | 0.082519 | 0.067048 |
| `11_crsid_basis_only_no_residual` | 30 | 28 | 0.061683 | 0.051820 |
| `12_crsid_no_semantic_basis` | 15 | 10 | 0.077641 | 0.064856 |
| `13_crsid_no_shared_residual` | 25 | 20 | 0.080821 | 0.064700 |
| `14_crsid_no_private_residual` | 30 | 28 | 0.072748 | 0.061424 |
| `15_crsid_fixed_alpha_050` | 17 | 12 | 0.079822 | 0.061736 |
| `16_crsid_private_only_alpha_100` | 17 | 12 | 0.078595 | 0.061063 |
| `17_crsid_shared_only_alpha_000` | 30 | 28 | 0.063200 | 0.055874 |
| `20_crsid_semhub_full_f005_g10_s10` | 17 | 12 | 0.078656 | 0.060424 |

完整 CRSID 的 best valid NDCG@10 最高，并且 test NDCG@10 也最高，说明不是单纯验证集偶然偏高。

`basis_only` 和 `shared_only` 训练到 30 epoch 才接近最佳，但 test 仍然很低，说明这些模型不是因为 early stopping 太早而失败，而是表达能力不足。

## 5. 推荐论文结论

可以将本轮消融总结为三点：

```text
1. CRSID full outperforms SASRec and QSDRec semantic-score baselines,
   validating the effectiveness of constructing item embeddings with Semantic IDs.

2. Removing semantic basis, shared semantic residual, private item residual,
   or adaptive alpha all degrades performance, confirming the necessity of each component.

3. Adaptive residual allocation is particularly important:
   fixed alpha, private-only residual, and shared-only residual all underperform the full model.
```

中文版本：

```text
完整 CRSID 在 Office 数据集上取得最高的 NDCG@10。相比纯 ID SASRec，NDCG@10 提升 14.04%；
相比 QSDRec 语义打分 baseline，NDCG@10 提升 11.10%。模块消融显示，移除 semantic basis、
shared semantic residual、private item residual 或自适应 alpha 都会造成性能下降。其中只保留
semantic basis 的版本下降最明显，说明协同残差是必要的；固定 alpha 和单一路径 residual 均明显
低于完整模型，说明根据 item 状态自适应分配 private/shared residual 是 CRSID 的关键设计。
```

## 6. 建议采用版本

建议当前主方法采用：

```text
10_crsid_full_tau20_s10
model_variant = crsid
cr_tail_tau = 20
cr_residual_scale = 1.0
num_hard_neg = 0
num_random_neg = 100
dim = 128
```

不建议将 `crsid_semhub` 作为主方法，因为它在本轮结果中低于 item-frequency alpha 版本：

```text
0.060424 vs 0.067048
```

但可以在论文中把 `crsid_semhub` 作为诊断消融或 negative ablation，说明并非所有 Semantic ID hubness 先验都能带来提升，当前数据上 item-level interaction frequency 是更稳定的 residual allocation signal。

## 7. 需要注意的限制

本轮结果已经足够支持模块有效性，但如果要写得更严谨，建议补充：

```text
1. 多 seed 重复实验，至少 seed = 2026, 2027, 2028。
2. 在 Beauty 数据集上复验主方法和关键消融。
3. 如果篇幅有限，论文主表保留 full / SASRec / QSDRec / no residual / no basis / no shared / no private / fixed alpha。
```

当前单 seed 结果趋势非常一致，可以作为方法选择依据；多 seed 主要用于增强论文可信度。

## 8. Beauty LC-SoftSID 必要消融结果

本节记录 Beauty 数据集上的 LC-SoftSID 必要消融结果，结果来自：

```text
runs/beauty/lc_soft_required_ablation/summary.csv
```

核心配置：

```text
dataset = Beauty
dim = 128
batch_size = 1024
num_random_neg = 100
num_hard_neg = 0
max_len = 50
lr = 0.001
weight_decay = 0.0001
seed = 2026
metric = full-ranking NDCG@10 / HR@10 / NDCG@20
```

需要特别说明：这一组是“必要消融”结果，主要用于比较模块相对贡献。当前最好主方法结果中有部分实验跑了更长训练，例如 100 epoch，而本轮消融中的若干实验实际训练轮数较少或 early stopping 条件不同。因此本节结果不应直接用于和 100 epoch 最优主结果做最终性能比较，而应主要作为模块趋势分析。

### 8.1 总体结果

按测试集 NDCG@10 排序：

| Rank | Experiment | NDCG@10 | HR@10 | NDCG@20 | HR@20 | Best Valid NDCG@10 |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `21_soft_no_local_pruning` | 0.051885 | 0.089881 | 0.061111 | 0.126593 | 0.062950 |
| 2 | `20_lc_soft_full` | 0.051396 | 0.089478 | 0.060372 | 0.125117 | 0.063337 |
| 3 | `10_hard_crsid` | 0.051285 | 0.090283 | 0.060418 | 0.126593 | 0.062951 |
| 4 | `22_soft_eta1_no_sharpen` | 0.051223 | 0.088897 | 0.060351 | 0.125117 | 0.063117 |
| 5 | `40_with_behavior_neighbors_w050` | 0.048582 | 0.086750 | 0.058052 | 0.124357 | 0.062953 |
| 6 | `30_no_shared_residual` | 0.047690 | 0.084515 | 0.057436 | 0.123195 | 0.062024 |
| 7 | `00_sasrec_id_only` | 0.044830 | 0.077047 | 0.052492 | 0.107454 | 0.058633 |
| 8 | `31_no_private_residual` | 0.034128 | 0.065063 | 0.042563 | 0.098556 | 0.045334 |

### 8.2 主要结论

#### 8.2.1 CRSID / LC-SoftSID 表示整体有效

Hard CRSID 相比纯 ID SASRec 有明显提升：

```text
Hard CRSID:     NDCG@10 = 0.051285
SASRec ID-only: NDCG@10 = 0.044830
Absolute gain:  +0.006455
Relative gain:  +14.40%
```

这说明在 Beauty 上，Semantic ID 相关的 item representation 明显优于纯 ID 序列推荐。该结果支持本文的基本动机：Semantic ID 的共享结构可以为序列推荐提供额外泛化能力。

#### 8.2.2 LC-SoftSID 相比 hard CRSID 有小幅正向提升

```text
LC-SoftSID full: NDCG@10 = 0.051396
Hard CRSID:      NDCG@10 = 0.051285
Absolute gain:   +0.000111
Relative gain:   +0.22%
```

本轮必要消融中，soft SID 相比 hard SID 的提升幅度较小，但方向为正。结合之前 Beauty soft tuning 中更长训练和更充分调参的结果，soft SID 方向仍然是有效的。不过本轮消融不能单独作为最终主性能结论。

#### 8.2.3 no local pruning 在本轮 Beauty 上最好

```text
No local pruning: NDCG@10 = 0.051885
LC-SoftSID full:  NDCG@10 = 0.051396
```

这说明 Beauty 上 `cr_soft_min_support=0.05` 可能偏保守。Beauty 商品语义较细碎，很多有效候选 token 在局部邻域中的支持度可能不高，严格 local pruning 可能剪掉一部分有用共享。

该结果不说明 local-consistent soft SID 的方向错误，而是说明 Beauty 上 soft candidate SID 的收益主要来自多候选表示和支持度加权；local pruning 阈值对数据集较敏感。论文表述应避免把 local pruning 说成在所有数据集上必然提升，而应写成：

```text
Local support is used to control candidate reliability. Its threshold can be tuned by validation, and overly strict pruning may remove useful fine-grained candidates on dense product domains.
```

中文可写为：

```text
局部支持度约束用于提高候选 SID 的可靠性，但其阈值需要结合数据集验证集选择。
在 Beauty 这类细粒度商品域中，过强的剪枝可能损失一部分有效候选 token。
```

#### 8.2.4 support sharpening 有小幅作用

对比：

```text
LC-SoftSID full eta=2: NDCG@10 = 0.051396
eta=1 no sharpen:      NDCG@10 = 0.051223
```

`support_eta=2.0` 比 `eta=1.0` 略好，说明局部支持度更高的 token 应被更强强调。这支持 soft SID 权重设计：

```text
s_{i,l}(c) = cnt_{i,l}(c)^\eta
```

不过该提升幅度较小，不应被夸大为主要性能来源。

#### 8.2.5 用户侧行为邻域当前不是正收益

行为邻域版本：

```text
Behavior neighbor w=0.5: NDCG@10 = 0.048582
LC-SoftSID full:         NDCG@10 = 0.051396
```

下降：

```text
-0.002814 NDCG@10
```

这说明当前基于训练序列窗口共现的 behavior neighbor 会给 Beauty 引入明显噪声。用户共现邻域可能把 item 拉向同一用户历史中的泛主题，而不一定是同系列或同语义功能 item，这与 semantic drift 的风险一致。

因此，当前最终主方法不采用 behavior neighbor。论文中如果保留“用户侧信息补充”的叙述，需要非常谨慎：它可以作为探索性扩展或 future work，而不应作为已经稳定验证的核心贡献。更稳妥的写法是：

```text
We further examine a train-only behavior-neighbor augmentation, but observe that naive co-occurrence neighbors introduce noise on Beauty. Therefore, the final model adopts the semantic local-consistency version.
```

#### 8.2.6 shared residual 有明显作用

去掉 shared semantic residual：

```text
LC-SoftSID full:       NDCG@10 = 0.051396
No shared residual:    NDCG@10 = 0.047690
Drop:                 -0.003706
Relative drop:        -7.21%
```

这说明 Semantic ID token 共享残差确实提供了有用的跨 item 泛化能力，尤其对于低频 item 和共享语义结构中的 item 有意义。

#### 8.2.7 private residual 是 Beauty 上最关键的模块

去掉 private item residual 后性能大幅下降：

```text
LC-SoftSID full:       NDCG@10 = 0.051396
No private residual:   NDCG@10 = 0.034128
Drop:                 -0.017268
Relative drop:        -33.60%
```

这是本轮 Beauty 消融中最强的模块证据。它说明不能只依赖 Semantic ID 共享。Beauty 中大量 item 具有细粒度差异，只用语义共享会严重损失 item-level 区分能力。private residual 对消除语义漂移和保留 item 个体差异非常关键。

该结果可以直接支撑方法设计中的 shared/private residual 分解：

```text
Shared semantic residual provides transfer across related items,
while private item residual preserves item-specific collaborative distinctions.
Both are necessary, and the private residual is especially important in fine-grained product domains.
```

### 8.3 与 100 epoch 最优主结果的关系

之前 Beauty soft tuning 中，当前较好的长训练配置为：

```text
27_crsid_soft_m4_s005_prior1_eta2_n50
NDCG@10 = 0.052575
HR@10   = 0.091177
NDCG@20 = 0.061586
```

而本轮必要消融中的 `20_lc_soft_full` 为：

```text
NDCG@10 = 0.051396
HR@10   = 0.089478
NDCG@20 = 0.060372
```

二者并不完全可比，因为训练轮数、early stopping 状态和实验目录不同。因此本节的 `20_lc_soft_full` 不应替代之前的 Beauty 最优主结果。合理使用方式是：

```text
1. 主表报告统一训练设置下的最终主方法结果。
2. 消融表报告同一组必要消融内部的相对趋势。
3. 不用不同 epoch / 不同 early stopping 的结果混合计算最终提升幅度。
```

这也是为什么本节更强调模块趋势，而不是声称 `21_soft_no_local_pruning` 就是最终 Beauty 主方法。

### 8.4 Beauty 消融小结

Beauty 必要消融可以总结为：

```text
1. Hard CRSID 和 LC-SoftSID 均明显优于纯 ID SASRec，说明 Semantic ID item representation 有效。
2. LC-SoftSID full 相比 hard CRSID 有小幅提升，但本轮消融不是最终主性能实验。
3. support_eta=2.0 略优于 eta=1.0，支持局部支持度加权。
4. local pruning 在 Beauty 上不是正收益，说明该阈值具有数据集敏感性。
5. behavior neighbor 当前引入噪声，不进入最终主方法。
6. shared residual 有明显贡献，private residual 是最关键模块。
```

论文写作中，建议将 Beauty 消融重点放在 residual 分解和 Semantic ID representation 的有效性上；对 local pruning 与 behavior neighbor 采用谨慎表述。
