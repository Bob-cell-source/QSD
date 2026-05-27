# EviQSD: Evidence-Calibrated Query-Guided Semantic Disambiguation

本文档以论文 Method 的形式描述当前代码中实际效果最好的版本，并补充数据泄露与实验可信性自查。当前最优配置来自 Office 数据集上的实际运行结果：

```text
num_interests = 8
sem_weight = 0.10
evidence_gate = history_overlap
evidence_floor = 0.20
num_random_neg = 100
num_hard_neg = 0
train_objective = sampled
valid/test = full-ranking evaluation
```

## 1. Problem Formulation

给定用户集合 \(\mathcal{U}\)、物品集合 \(\mathcal{I}\)，以及用户 \(u\) 的历史交互序列：

\[
S_u = [i_1, i_2, \ldots, i_t],
\]

目标是在时间步 \(t+1\) 对所有候选物品进行排序，并将真实下一个交互物品 \(i^+\) 排在尽可能靠前的位置。模型输出用户 \(u\) 对候选物品 \(i\) 的预测分数：

\[
s(u,i).
\]

训练阶段使用采样负例构造小批量分类任务；验证和测试阶段对全量 item 做 ranking evaluation，并报告 HR@K、Recall@K 和 NDCG@K，其中 \(K \in \{5,10,20\}\)。

## 2. Item Semantic Tokenization

每个物品 \(i\) 具有文本侧信息，例如标题、品牌、类别、描述等。首先使用文本编码器 \(g(\cdot)\) 将物品元数据编码为连续语义向量：

\[
x_i = g(\text{text}_i) \in \mathbb{R}^d.
\]

当前实验使用：

```text
encoder = BAAI/bge-small-en-v1.5
codebook_sizes = 64,128,256,512
```

随后使用 Residual Quantization K-Means 将连续语义向量离散化为长度为 \(F\) 的 Semantic ID：

\[
z_i = [z_{i,1}, z_{i,2}, \ldots, z_{i,F}],
\]

其中当前实现中 \(F=4\)。每个 \(z_{i,f}\) 是第 \(f\) 个语义槽位上的离散 code。该离散结构使不同 item 可以通过共享 Semantic ID token 来共享语义统计信号，尤其有利于低频 item。

## 3. Sequential ID Branch

ID 分支采用 SASRec 编码用户历史交互序列。给定用户历史 \(S_u\)，模型得到用户当前协同表示：

\[
h_u = \text{SASRec}(S_u).
\]

候选物品 \(i\) 的 ID embedding 为 \(e_i\)，ID 分支打分为：

\[
s_{\text{id}}(u,i) = h_u^\top e_i.
\]

该分支主要负责建模协同行为模式、序列转移模式和 item-level 记忆能力。

## 4. Query-Guided Semantic Branch

语义分支将用户历史中的 item Semantic ID 转换为语义记忆，并结合 ID 分支输出的用户表示生成多个语义兴趣查询：

\[
Q_u = [q_{u,1}, q_{u,2}, \ldots, q_{u,K}],
\]

其中 \(K\) 为语义兴趣 query 数量。当前最优实验中：

```text
K = num_interests = 8
```

候选物品 \(i\) 的每个 Semantic ID token \(z_{i,f}\) 被映射为语义 token embedding：

\[
t_{i,f} = E_z(z_{i,f}).
\]

原始 QSD 语义分支会对 query 和候选语义 token 进行 token-level attention：

\[
\alpha_{u,i,k,f}
=
\frac{
\exp(q_{u,k}^{\top} t_{i,f} / \sqrt{d})
}{
\sum_{f'=1}^{F}
\exp(q_{u,k}^{\top} t_{i,f'} / \sqrt{d})
}.
\]

然后得到 query-specific 的候选语义响应：

\[
r_{u,i,k}
=
\sum_{f=1}^{F}
\alpha_{u,i,k,f} t_{i,f}.
\]

最后通过 query router 聚合多个兴趣 query 的匹配分数，得到语义分支分数：

\[
s_{\text{sem}}(u,i)
=
\sum_{k=1}^{K}
\beta_{u,i,k}
\cdot
q_{u,k}^{\top} r_{u,i,k}.
\]

其中 \(\beta_{u,i,k}\) 是候选感知的 query 权重。

## 5. Semantic Evidence Calibration

实验分析发现，Semantic ID 共享既能带来泛化能力，也会造成语义漂移。尤其在低频 item 或高共享语义簇中，模型容易被热门 semantic token、通用类别 token 或 dominant history theme 吸引，导致 target 被同类但不等价的 item 挤出 Top-K。

因此，当前最优版本引入 Semantic Evidence Calibration。核心思想是：候选物品的 Semantic ID token 不应默认全部可信，而应根据用户历史中是否存在同槽位语义证据进行校准。

对候选 item \(i\) 的第 \(f\) 个语义 token \(z_{i,f}\)，定义用户历史证据：

\[
c_{u,i,f}
=
\mathbb{I}
\left[
\exists j \in S_u,
z_{j,f} = z_{i,f}
\right].
\]

即如果用户历史中某个 item 在相同 slot \(f\) 上与候选 item 共享同一个 Semantic ID token，则认为该 token 被用户历史支持。

进一步定义 evidence weight：

\[
w_{u,i,f}
=
\begin{cases}
1, & c_{u,i,f}=1, \\
\rho, & c_{u,i,f}=0,
\end{cases}
\]

其中 \(\rho\) 是 evidence floor。当前最优配置为：

\[
\rho = 0.20.
\]

evidence floor 的作用是避免过度过滤未在历史中出现过的语义 token，使模型仍然保留一定的语义泛化能力。

随后，原始 token-level attention 被 evidence weight 校准：

\[
\tilde{\alpha}_{u,i,k,f}
=
\frac{
\alpha_{u,i,k,f} \cdot w_{u,i,f}
}{
\sum_{f'=1}^{F}
\alpha_{u,i,k,f'} \cdot w_{u,i,f'}
}.
\]

校准后的语义响应为：

\[
\tilde{r}_{u,i,k}
=
\sum_{f=1}^{F}
\tilde{\alpha}_{u,i,k,f} t_{i,f}.
\]

最终语义分支分数变为：

\[
s_{\text{sem}}^{\text{evi}}(u,i)
=
\sum_{k=1}^{K}
\beta_{u,i,k}
\cdot
q_{u,k}^{\top} \tilde{r}_{u,i,k}.
\]

该机制不是对整个 semantic branch 做一个全局 gate，而是在 candidate Semantic ID token level 上判断每个语义 token 是否被当前用户历史支持。它保留了 Semantic ID 的共享泛化能力，同时抑制缺乏历史证据支持的 semantic shortcut。

## 6. Final Prediction

模型最终融合 ID 分支和 evidence-calibrated semantic branch：

\[
s(u,i)
=
s_{\text{id}}(u,i)
+
\lambda
s_{\text{sem}}^{\text{evi}}(u,i).
\]

当前最优配置为：

\[
\lambda = 0.10.
\]

因此，语义分支不是替代 ID 分支，而是作为轻量的语义校正项补充协同 ID 表示。

## 7. Training Objective

训练时对每个正样本 \(i^+\) 采样 \(M\) 个随机负样本：

\[
\mathcal{C}_u = \{i^+\} \cup \{i_1^-, \ldots, i_M^-\}.
\]

当前最优配置为：

```text
num_random_neg = 100
num_hard_neg = 0
```

模型对候选集合 \(\mathcal{C}_u\) 计算分数，并使用 softmax cross-entropy 优化：

\[
\mathcal{L}
=
-
\log
\frac{
\exp(s(u,i^+))
}{
\sum_{i \in \mathcal{C}_u}
\exp(s(u,i))
}.
\]

验证和测试时不使用采样候选，而是对全量 item 做排序评估，从而保证模型选择和最终测试协议一致。

## 8. Method Summary

EviQSD 的关键设计可以概括为：

1. 使用 SASRec 作为 ID 协同序列建模 backbone。
2. 使用 RQ-KMeans Semantic ID 为 item 构建多槽位离散语义表示。
3. 使用 query-guided semantic branch 建模用户多语义兴趣与候选 Semantic ID token 的匹配。
4. 使用 Semantic Evidence Calibration 在 token-level 上判断候选语义 token 是否被用户历史同槽位 token 支持。
5. 通过 ID score 与 evidence-calibrated semantic score 融合进行最终推荐。

一句话描述：

> EviQSD calibrates each candidate Semantic ID token according to whether it is supported by the user's historical semantic evidence, enabling reliable semantic sharing while suppressing unsupported semantic shortcuts.

## 9. Data Leakage and Experimental Validity Audit

本节自查当前实现是否存在数据泄露、不公平 trick 或可能影响可信性的风险。

### 9.1 Interaction Split

当前推荐训练使用用户历史序列构造训练样本，验证和测试分别使用后续 held-out item 评估。验证和测试阶段采用全量 item ranking，而不是 sampled evaluation。这一点避免了 sampled negative evaluation 带来的指标虚高问题。

可信性判断：

```text
通过。valid/test 使用全量排序，比 sampled test 更严格。
```

需要在论文中明确：

```text
For each user, we use historical interactions for training and hold out subsequent interactions for validation and testing. During validation and testing, all items are ranked except items that should be masked according to the evaluation protocol.
```

### 9.2 Semantic ID Construction

当前 Semantic ID 使用 item metadata 构建，包括 title、brand、category、description 等 item-side text，不使用用户交互标签、不使用 valid/test target label，也不使用模型测试结果。

可信性判断：

```text
基本通过，但属于 transductive item-feature setting。
```

原因是：RQ-KMeans codebook 通常是在当前数据集的全部 item metadata 上训练的，其中包括 valid/test 中可能出现的 item。由于这些 item metadata 是推荐系统实际可用的 item 侧静态信息，不包含用户未来行为标签，因此一般不视为交互泄露。但论文必须明确这是 item metadata available setting。

建议论文写法：

```text
Semantic IDs are constructed only from item-side textual metadata and do not use user-item interaction labels from validation or test sets.
```

如果审稿人要求严格 inductive setting，则需要额外实验：

```text
只用 train item 训练 codebook，然后将 valid/test-only item 通过最近 codeword assignment 编码。
```

当前版本未做这个更严格设置，因此不要声称是严格 inductive cold-start。

### 9.3 Evidence Calibration 是否看到了 target

Evidence Calibration 的输入是当前用户历史序列 \(S_u\) 和候选 item \(i\) 的 Semantic ID。对每个候选 item 都会计算：

\[
\exists j \in S_u, z_{j,f}=z_{i,f}.
\]

它没有使用 target label 是否正确，也没有使用 valid/test future interaction。测试时每个候选 item 都按相同规则计算 evidence，因此不属于 label leakage。

可信性判断：

```text
通过。Evidence 是 candidate-conditioned feature，不是 target-only feature。
```

需要注意的表述：

```text
不要写成“根据真实 target 的共享情况调整分数”。
应该写成“for each candidate item, we compute whether its semantic tokens are supported by the user's observed history”。
```

### 9.4 Full-Ranking Evaluation

当前 valid/test 使用全量 item 计算 HR/Recall/NDCG@5/10/20，并且 best model 根据 full-ranking valid NDCG@10 选择。测试也使用同一套 full-ranking 逻辑。

可信性判断：

```text
通过。验证选择和测试协议一致。
```

这比“验证 sampled、测试 full-ranking”更可信。

### 9.5 Negative Sampling

当前最优版本训练使用随机负采样，而不是全量 softmax。验证和测试是全量 ranking。训练 sampled、测试 full-ranking 是序列推荐中常见做法，不构成 trick。

可信性判断：

```text
通过，但需要如实报告训练目标为 sampled softmax/cross-entropy。
```

### 9.6 Hyperparameter Selection

当前 evidence floor、sem weight、num interests 是通过 Office 数据集实验选择出来的。若在 Beauty 上直接复用这些参数，再报告 Beauty 结果，则 Beauty 可以作为更可信的迁移验证。如果在 Beauty 上继续大量调参，则需要说明验证集调参过程。

可信性判断：

```text
需要规范报告。
```

建议：

1. 主结果中使用验证集选择最优模型 checkpoint。
2. 超参数选择只基于 valid NDCG@10。
3. 不要根据 test 指标反复改方法后只报告最好 test。
4. 对最终方法至少跑 3 个 seed，报告 mean ± std。

### 9.7 Preprocessing Filter

Amazon 推荐实验通常会使用 k-core filtering，例如 `min_user_inter=5`、`min_item_inter=5`。这种 filtering 往往基于全量交互数据完成，因此严格来说属于 transductive dataset construction，但这是该类 benchmark 的常见做法。

可信性判断：

```text
可接受，但需要报告 preprocessing protocol。
```

建议论文写法：

```text
We apply 5-core filtering following common practice in sequential recommendation.
```

不要将该设置描述为完全在线或严格时间隔离的数据构建。

### 9.8 当前最需要补强的可信性实验

为了让结果更真实可信，建议补充：

1. `3 seeds`：例如 `2024, 2025, 2026`。
2. `Office + Beauty` 两个数据集均报告。
3. 与 SASRec、QSD-base、EviQSD 做同设置对比。
4. 报告 full-ranking 指标，不使用 sampled test。
5. 补充分组结果：low-frequency、prefix sharing、slot overlap sharing。
6. 补充 badcase，展示 EviQSD 成功和失败的真实样本。

### 9.9 总体可信性结论

当前结果没有明显 label leakage 或 sampled-evaluation trick。需要注意的是，Semantic ID 构建使用全量 item metadata，因此应明确为 item-side metadata transductive setting；这在推荐系统中通常是合理的，但不应包装成严格 inductive cold-start。

更严谨的论文结论应该是：

```text
EviQSD improves sequential recommendation under a full-ranking evaluation protocol by calibrating candidate Semantic ID tokens with user historical semantic evidence. The method uses only item-side metadata and observed user history, without validation/test interaction labels during training or scoring.
```

