# R-EviQSD: Reliability-aware Evidence-Calibrated QSDRec

本文档整理当前最适合论文叙事的方法版本、模型公式、实验结论和可信性说明。当前推荐主方法命名为：

```text
R-EviQSD: Reliability-aware Evidence-Calibrated Query-guided Semantic Disambiguation
```

核心观点：

```text
Semantic ID sharing is useful but unreliable.
Not every shared semantic token is valid intent evidence; some are semantic shortcuts or hubs.
```

因此，R-EviQSD 不再默认所有 Semantic ID token 都同等可信，而是为候选 item 的每个 semantic token 估计 token-level evidence reliability，并用该可靠性校准 semantic attention。

## 1. Motivation

在基于 Semantic ID 的序列推荐中，item 被表示为多槽位离散语义码：

\[
z_i = [z_{i,1}, z_{i,2}, \ldots, z_{i,F}].
\]

Semantic ID 的优势是可以让低频 item 通过共享语义 token 获得泛化能力。例如，对于低频墨盒、文件夹、办公用品，如果用户历史中出现过同系列或同类别商品，semantic branch 可以把目标 item 拉入候选前列。

但实验和 badcase 也显示，直接使用 Semantic ID 会带来两类问题：

1. **Semantic under-sharing**：真实相关 item 可能没有完全共享 prefix 或 exact token，导致低频 target 仍然得不到足够语义支持。
2. **Semantic over-sharing / hubness**：热门语义 token 或通用办公主题会成为 shortcut，把推荐结果拉向 marker、binder、printer、label 等高密度语义簇。

因此，问题不只是“是否使用语义”，而是：

```text
Which semantic tokens are reliable evidence for the current user-candidate pair?
```

## 2. Problem Formulation

给定用户 \(u\) 的历史序列：

\[
S_u = [i_1, i_2, \ldots, i_t],
\]

目标是在候选全集 \(\mathcal{I}\) 中对下一个 item \(i^+\) 排序。模型输出：

\[
s(u,i)
\]

并在验证和测试阶段使用 full-ranking HR@K、Recall@K、NDCG@K 评估，其中 \(K \in \{5,10,20\}\)。

## 3. Item Semantic Tokenization

每个 item 的标题、品牌、类别、描述等元数据首先由文本编码器编码为连续向量：

\[
x_i = g(\text{text}_i).
\]

当前实验使用：

```text
encoder = BAAI/bge-small-en-v1.5
codebook_sizes = 64,128,256,512
```

然后通过 RQ-KMeans 将连续语义向量量化为长度为 \(F=4\) 的 Semantic ID：

\[
z_i = [z_{i,1}, z_{i,2}, z_{i,3}, z_{i,4}].
\]

该 Semantic ID 仅由 item-side metadata 构建，不使用 valid/test 交互标签。

## 4. Sequential ID Branch

ID 分支采用 SASRec 编码用户历史序列：

\[
h_u = \text{SASRec}(S_u).
\]

候选 item \(i\) 的 ID embedding 为 \(e_i\)，ID 分支分数为：

\[
s_{\text{id}}(u,i)=h_u^\top e_i.
\]

该分支负责协同过滤、序列转移和 item-level 记忆。

## 5. Query-guided Semantic Branch

Semantic branch 将用户历史中的 Semantic ID 映射为语义记忆，并生成 \(K\) 个用户语义 query：

\[
Q_u=[q_{u,1},q_{u,2},\ldots,q_{u,K}].
\]

候选 item \(i\) 的第 \(f\) 个 Semantic ID token 被映射为 embedding：

\[
t_{i,f}=E_z(z_{i,f}).
\]

原始 QSD 对 query 和候选 token 做 token-level attention：

\[
\alpha_{u,i,k,f}
=
\frac{
\exp(q_{u,k}^{\top}t_{i,f}/\sqrt{d})
}{
\sum_{f'=1}^{F}
\exp(q_{u,k}^{\top}t_{i,f'}/\sqrt{d})
}.
\]

得到 query-specific semantic response：

\[
r_{u,i,k}
=
\sum_{f=1}^{F}
\alpha_{u,i,k,f}t_{i,f}.
\]

再由 query router 聚合多个 query：

\[
s_{\text{sem}}(u,i)
=
\sum_{k=1}^{K}
\beta_{u,i,k}
q_{u,k}^{\top}r_{u,i,k}.
\]

原始 QSD 的问题是：\(\alpha\) 只由 query-token similarity 决定，没有判断 token 是否被当前用户历史支持。

## 6. Token-level Evidence Reliability

R-EviQSD 为候选 item 的每个 semantic token 估计 reliability：

\[
\omega_{u,i,f} \in [\rho,1].
\]

其中 \(\rho\) 是 evidence floor，用于避免完全丢弃无历史覆盖的 semantic token。当前主配置：

```text
evidence_floor = 0.20
```

### 6.1 Evidence Features

对每个候选 token \(z_{i,f}\)，构造用户相关证据特征：

#### Same-slot Evidence Strength

检查历史 item 是否在相同 slot 上共享 token：

\[
c^{same}_{u,i,f}
=
\sum_{j \in S_u}
\mathbb{I}(z_{j,f}=z_{i,f})
\cdot
\gamma(j),
\]

其中 \(\gamma(j)\) 是 recency weight，越靠近当前时间的历史 item 权重越高。代码中使用饱和函数：

\[
\hat{c}^{same}_{u,i,f}=1-\exp(-c^{same}_{u,i,f}).
\]

#### Cross-slot Auxiliary Evidence

为缓解 prefix-only 或 fixed-slot matching 漏掉真实邻居的问题，引入小权重跨槽位辅助证据：

\[
c^{cross}_{u,i,f}
=
\sum_{j \in S_u}
\sum_{g \neq f}
\mathbb{I}(z_{j,g}=z_{i,f})
\cdot
\gamma(j).
\]

由于 RQ-KMeans 各槽位 codebook 不完全等价，cross-slot evidence 只作为辅助特征，不直接替代 same-slot evidence。当前主配置：

```text
evidence_cross_weight = 0.20
```

#### Token Specificity

为刻画 semantic hubness，统计每个 token 在全局 item 集中的频率，并得到归一化 hubness \(h_f(k)\)。token specificity 定义为：

\[
spec_f(k)=1-h_f(k).
\]

高频 token specificity 低，说明它更可能是泛化语义或 shortcut；低频 token specificity 高，说明它更可能携带具体语义。

#### Latest Support

记录最近一次支持该 token 的历史位置：

\[
p^{latest}_{u,i,f}.
\]

这用于区分近期兴趣证据和很久以前的弱证据。

### 6.2 Reliability Estimator

R-EviQSD 使用轻量 evidence reliability estimator：

\[
r_{u,i,f}
=
\sigma(
\text{MLP}(
[
\hat{c}^{same}_{u,i,f},
\hat{c}^{cross}_{u,i,f},
spec_f(z_{i,f}),
p^{latest}_{u,i,f}
]
)).
\]

最终 token evidence weight 为：

\[
\omega_{u,i,f}
=
\rho+(1-\rho)r_{u,i,f}.
\]

该模块不是普通 fusion gate。它不直接控制整个 semantic branch，而是在 candidate Semantic ID token level 判断每个 token 是否是当前用户-候选对的可靠语义证据。

## 7. Evidence-Calibrated Semantic Attention

使用 reliability weight 校准原始 token attention：

\[
\tilde{\alpha}_{u,i,k,f}
=
\frac{
\alpha_{u,i,k,f}\omega_{u,i,f}
}{
\sum_{f'=1}^{F}
\alpha_{u,i,k,f'}\omega_{u,i,f'}
}.
\]

校准后的 semantic response：

\[
\tilde{r}_{u,i,k}
=
\sum_{f=1}^{F}
\tilde{\alpha}_{u,i,k,f}t_{i,f}.
\]

最终 semantic score：

\[
s_{\text{sem}}^{rel}(u,i)
=
\sum_{k=1}^{K}
\beta_{u,i,k}
q_{u,k}^{\top}
\tilde{r}_{u,i,k}.
\]

## 8. Final Prediction

最终推荐分数由 ID branch 和 reliability-calibrated semantic branch 融合：

\[
s(u,i)
=
s_{\text{id}}(u,i)
+
\lambda s_{\text{sem}}^{rel}(u,i).
\]

当前主配置：

```text
sem_weight = 0.10
```

Beauty 上发现 \(K=4\) 有时优于 \(K=8\)，说明多兴趣数量应根据验证集选择。论文中可将 `num_interests` 作为验证集选择的超参数。

## 9. Training and Evaluation

训练阶段使用 sampled softmax：

\[
\mathcal{C}_u=\{i^+\}\cup\{i_1^-,\ldots,i_M^-\},
\]

其中当前配置：

```text
num_random_neg = 100
num_hard_neg = 0
```

训练损失：

\[
\mathcal{L}
=
-
\log
\frac{
\exp(s(u,i^+))
}{
\sum_{i\in\mathcal{C}_u}
\exp(s(u,i))
}.
\]

验证和测试阶段均使用 full-ranking evaluation，即真实 item 与全量 item 比较，并使用 full-ranking validation NDCG@10 选择 best checkpoint。

论文表述：

```text
During training, we optimize a sampled softmax objective with randomly sampled negative items.
During validation and testing, we rank the ground-truth item against the full item set and select the best checkpoint according to full-ranking validation NDCG@10.
```

## 10. Experimental Findings

### 10.1 Office Controlled Probe

Office reliability ablation 中，Learnable Reliability 最优：

| Method | NDCG@10 | HR@10 | NDCG@20 |
|---|---:|---:|---:|
| Learnable Reliability | 0.062506 | 0.111723 | 0.077348 |
| Binary + Hub Penalty | 0.061983 | 0.113354 | 0.075331 |
| Binary Evidence | 0.061503 | 0.112538 | 0.075109 |
| QSD-base | 0.059813 | 0.109072 | 0.074839 |
| SASRec | 0.060193 | 0.106830 | 0.073461 |

分组结果显示，Learnable Reliability 相对 QSD-base 在所有 overlap group 上基本正向；相对 Binary Evidence，主要提升中等 evidence 区间，例如 `6-10` 和 `21-50`。

解释：

```text
Binary evidence is strong when exact evidence is reliable.
Learnable reliability is more useful when evidence exists but its reliability is ambiguous.
```

### 10.2 Beauty Reliability Full

Beauty 服务器结果显示，原始 QSD-base 不稳定，甚至略低于 SASRec：

| Method | Seed | NDCG@10 | HR@10 | NDCG@20 |
|---|---:|---:|---:|---:|
| SASRec | 2026 | 0.041873 | 0.074990 | 0.049920 |
| QSD-base | 2026 | 0.041742 | 0.071815 | 0.049813 |
| Binary Evidence | 2026 | 0.042541 | 0.073470 | 0.050468 |
| Binary + Hub Penalty | 2026 | 0.043192 | 0.074140 | 0.051446 |
| Learnable Reliability K=8 | 2026 | 0.042252 | 0.073425 | 0.050314 |
| Learnable Reliability K=4 | 2026 | 0.043281 | 0.077405 | 0.051494 |

三种子平均：

```text
Binary Evidence:
2024: 0.041172
2025: 0.043634
2026: 0.042541
mean ≈ 0.042449

Learnable Reliability K=8:
2024: 0.043265
2025: 0.043576
2026: 0.042252
mean ≈ 0.043031
```

Beauty 结论：

1. Naive QSD 不稳定，说明直接加入 Semantic ID 不一定有效。
2. Evidence Calibration 能稳定改善 QSD-base。
3. Hub Penalty 的提升说明热门 semantic token 的确会造成干扰。
4. Learnable Reliability 多 seed 平均更高，说明自适应 token reliability 比固定 binary rule 更稳。

## 11. Badcase Summary

### Learnable Reliability 修复的样本

典型包括：

```text
Columbian catalog envelopes
Pendaflex file folders
Quartet dry-erase board
Scotch tape
```

这些样本通常有中等强度的历史语义证据，但原始 QSD 容易被泛办公主题带偏。Learnable Reliability 能更好地区分真正支持 target 的 evidence 和 semantic shortcut。

### Binary Evidence 更稳的样本

典型包括：

```text
Wilson Jones binder
Brother toner cartridge
HP 940XL ink cartridge
```

这些样本存在强 exact evidence 或同系列历史 item，Binary Evidence 已经足够可靠。Learnable Reliability 有时会过度泛化到品牌生态或相邻类别。

### 共同失败样本

两者都错的样本平均训练频次较低，说明低频且语义证据不稳定的 target 仍是主要难点。

## 12. Data Leakage and Validity Check

当前实验设置没有明显 label leakage 或 sampled-evaluation trick：

1. Semantic ID 仅使用 item-side metadata 构建，不使用 valid/test interaction labels。
2. Evidence reliability 对每个候选 item 统一计算，不只对 ground-truth target 计算，因此不是 target leakage。
3. Valid/test 均使用 full-ranking evaluation。
4. Best checkpoint 根据 full-ranking valid NDCG@10 选择。
5. 训练 sampled softmax、验证测试 full-ranking 是常见设置，但需要如实报告。

需要明确的边界：

```text
Semantic IDs are constructed under an item-metadata-available setting.
This is not a strict inductive cold-start setting.
```

## 13. Recommended Paper Claims

建议论文主张控制在以下范围：

1. Directly introducing Semantic IDs is not consistently beneficial due to semantic shortcuts and token-level hubness.
2. User-history-supported evidence is necessary to make Semantic ID sharing reliable.
3. R-EviQSD estimates token-level evidence reliability and calibrates semantic attention accordingly.
4. The method improves over QSD-base and shows better stability across seeds, especially on Beauty.
5. Binary Evidence is a strong interpretable baseline, while Learnable Reliability provides better average robustness.

推荐英文摘要句：

```text
We propose R-EviQSD, a reliability-aware evidence calibration framework for Semantic ID based sequential recommendation. Instead of treating all semantic tokens as equally informative, R-EviQSD estimates token-level evidence reliability from user-specific historical support and global token specificity, and uses it to calibrate semantic attention. This enables the model to exploit reliable semantic sharing while suppressing unsupported semantic shortcuts.
```

推荐中文摘要句：

```text
本文提出 R-EviQSD，一种面向 Semantic ID 序列推荐的可靠性语义证据校准方法。该方法不再默认所有语义 token 均同等可信，而是根据用户历史支持、跨槽位辅助证据、token 特异性和近期性估计候选语义 token 的可靠性，并据此校准语义注意力，从而利用可靠语义共享并抑制无支撑的语义 shortcut。
```

