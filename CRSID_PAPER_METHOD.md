# CRSID 方法章节草稿

## Collaborative-Residual Semantic ID Representation

为充分利用 Semantic ID 中包含的语义结构，同时避免将语义信息仅作为额外打分分支引入，本文提出 Collaborative-Residual Semantic ID Representation，简称 CRSID。CRSID 的核心思想是将 item 表示分解为语义基底和协同残差两部分，使 Semantic ID 直接参与 item embedding 的构造过程，而不是在最终打分阶段额外叠加 semantic score。

给定 item \(i\) 的 Semantic ID：

\[
z_i = [z_{i,1}, z_{i,2}, \ldots, z_{i,L}],
\]

其中 \(L\) 表示 Semantic ID 的槽位数量。CRSID 首先为 Semantic ID token 构造语义基底表示。具体地，模型使用一张 Semantic ID embedding 表 \(E_b\)，对 item 的多个 Semantic ID token 进行平均池化，并经过线性投影得到 semantic basis：

\[
b_i = W_b \cdot \operatorname{Pool}(E_b(z_{i,1}), E_b(z_{i,2}), \ldots, E_b(z_{i,L})).
\]

该语义基底表示 item 在语义空间中的基础位置。共享相似 Semantic ID 的 item 可以通过该部分获得相近的初始表示，从而缓解纯 ID embedding 在长尾 item 上学习不足的问题。

仅依赖语义基底并不足以表达用户交互中的协同过滤信号。因此，CRSID 进一步引入协同残差。协同残差由两部分组成：item 私有残差和 Semantic ID 共享残差。item 私有残差由 item-level embedding 表得到：

\[
r_i^{p} = E_p(i),
\]

用于建模 item 自身独有的协同过滤偏移。Semantic ID 共享残差由另一张 Semantic ID token embedding 表 \(E_s\) 得到：

\[
r_i^{s} = \operatorname{Pool}(E_s(z_{i,1}), E_s(z_{i,2}), \ldots, E_s(z_{i,L})).
\]

其中 \(r_i^{s}\) 在共享 Semantic ID token 的 item 之间复用，用于传递语义相近 item 的协同信息，尤其适合交互稀疏的长尾 item。

为了在 item 私有协同信息和语义共享协同信息之间自适应平衡，CRSID 为每个 item 定义混合系数 \(\alpha_i\)：

\[
r_i = \alpha_i r_i^{p} + (1 - \alpha_i) r_i^{s}.
\]

在主版本中，\(\alpha_i\) 由 item 在训练集中的交互频次决定：

\[
\alpha_i = \frac{f_i}{f_i + \tau},
\]

其中 \(f_i\) 表示 item \(i\) 在训练历史中的出现次数，\(\tau\) 是平滑超参数。该设计体现了 memorization-generalization trade-off：对于交互频次较高的头部 item，\(\alpha_i\) 较大，模型更多依赖 item 私有残差以保留其独特协同模式；对于交互较少的长尾 item，\(\alpha_i\) 较小，模型更多依赖 Semantic ID 共享残差，从语义相近 item 中获得可迁移的协同信号。

最终，CRSID 的 item 表示定义为：

\[
e_i = \operatorname{LayerNorm}\left(
b_i + \lambda r_i
\right),
\]

即：

\[
e_i = \operatorname{LayerNorm}\left(
b_i + \lambda \left[
\alpha_i r_i^{p} + (1 - \alpha_i) r_i^{s}
\right]
\right),
\]

其中 \(\lambda\) 为 residual scale，用于控制协同残差相对于语义基底的强度。

在序列建模阶段，CRSID 使用动态 item 表示替代传统 SASRec 中的静态 item embedding。给定用户历史序列：

\[
S_u = [i_1, i_2, \ldots, i_t],
\]

模型首先为序列中的每个 item 构造 CRSID 表示：

\[
[e_{i_1}, e_{i_2}, \ldots, e_{i_t}],
\]

然后将其输入 causal Transformer encoder，得到用户表示：

\[
h_u = \operatorname{SASRecEncoder}(e_{i_1}, e_{i_2}, \ldots, e_{i_t}).
\]

除 item embedding 来源不同外，序列编码器保持与 SASRec 一致，包括 position embedding、causal self-attention、feed-forward network 和 layer normalization。

对于候选 item \(i\)，模型同样使用 CRSID 构造其表示 \(e_i\)，并通过点积计算推荐分数：

\[
\hat{y}_{u,i} = h_u^\top e_i.
\]

因此，CRSID 不再引入额外 semantic score、evidence gate、hub penalty 或 contrastive branch。Semantic ID 的作用已经被整合进 item representation 本身。

训练时，模型采用 sampled softmax objective。对于每个训练样本，候选集合由一个正样本和若干负样本组成：

\[
\mathcal{C}_u = \{i^+, i_1^-, \ldots, i_K^-\}.
\]

模型对候选集合计算分数，并优化交叉熵损失：

\[
\mathcal{L}_{rec}
= - \log
\frac{\exp(\hat{y}_{u,i^+})}
{\sum_{j \in \mathcal{C}_u} \exp(\hat{y}_{u,j})}.
\]

可选地，可以对 item 私有残差加入 \(L_2\) 正则：

\[
\mathcal{L}
=
\mathcal{L}_{rec}
+
\beta \lVert r_i^p \rVert_2^2,
\]

但主实验中默认不启用该正则。

综上，CRSID 通过语义基底、共享语义残差、私有 item 残差和频次自适应混合系数，将 Semantic ID 的语义泛化能力与协同过滤的 item-specific 表达能力结合起来。该方法保持了判别式序列推荐的简单打分形式，同时缓解了纯 ID 表示对长尾 item 学习不足、以及纯语义共享表示难以保留头部 item 个性化协同模式的问题。
