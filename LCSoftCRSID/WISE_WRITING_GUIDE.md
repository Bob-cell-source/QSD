# WISE 推荐系统论文写作指南

## 1. 会议与出版格式

目标会议为 International Conference on Web Information Systems Engineering（WISE）。近期主会论文由 Springer Lecture Notes in Computer Science（LNCS）出版。WISE 2024 的正式论文集包括 LNCS 15436–15440，其中 Part III（LNCS 15438）设置了独立的 **Recommendation Systems** 分区。

论文排版应直接使用 Springer 官方 LNCS 模板，不自行修改页边距、字号、标题和参考文献样式。具体投稿页数和匿名要求属于每届会议规则，必须以目标年份的官方 Call for Papers 为准，不能仅根据已出版论文页数推断。

Springer 官方模板与说明：

- [Springer Computer Science Proceedings Guidelines](https://www.springer.com/gp/computer-science/lncs/conference-proceedings-guidelines)
- [WISE conference series](https://link.springer.com/conference/wise)
- [WISE 2024 Recommendation Systems proceedings](https://dblp.org/db/conf/wise/wise2024-3.html)

## 2. 近期推荐系统论文

WISE 2024 的 Recommendation Systems 分区包含多模态推荐、多行为推荐、跨域推荐、序列推荐、新闻推荐和图推荐等工作。与 LC-SOFT CRSID 最接近的写作参考为：

1. [Cross-Domain Sequential Recommendation with Temporal Encoding and Projection-Based Learning](https://link.springer.com/chapter/10.1007/978-981-96-0570-5_6)，WISE 2024，pp. 75–90。
2. [The Research of Sequence Recommendation Method Based on Heterogeneous Enhanced Transformer with Multi-behavior Data](https://link.springer.com/chapter/10.1007/978-981-96-0570-5_11)，WISE 2024，pp. 148–163。
3. [MDAP: A Multi-view Disentangled and Adaptive Preference Learning Framework for Cross-Domain Recommendation](https://ar5iv.org/abs/2410.05877)，WISE 2024，pp. 164–178。
4. [Causal Behavior Pattern Inference for News Recommendation Through Multi-interest Matching](https://link.springer.com/chapter/10.1007/978-981-96-0570-5_13)，WISE 2024，pp. 179–190。
5. [MIN: Multi-stage Interactive Network for Multimodal Recommendation](https://link.springer.com/chapter/10.1007/978-981-96-0570-5_14)，WISE 2024，pp. 191–205。
6. [MHHCR: Multi-behavior Heterogeneous Hypergraph Contrastive Recommendation](https://link.springer.com/chapter/10.1007/978-981-96-0570-5_7)，WISE 2024，pp. 91–102。
7. [Self-attention Convolutional Neural Network for Sequential Recommendation](https://link.springer.com/chapter/10.1007/978-981-99-7254-8_44)，WISE 2023，pp. 569–578。
8. [Informative Anchor-Enhanced Heterogeneous Global Graph Neural Networks for Personalized Session-Based Recommendation](https://link.springer.com/chapter/10.1007/978-981-99-7254-8_45)，WISE 2023，pp. 579–593。

## 3. WISE 写作特征

### 3.1 摘要

近期 WISE 推荐论文的摘要通常采用紧凑的五步结构：

1. 一至两句说明推荐任务和应用价值；
2. 明确指出现有方法的一个或两个具体缺陷；
3. 给出方法名称和核心思想；
4. 按执行顺序概括两至三个主要模块；
5. 用数据集和主要结论收尾。

摘要强调问题、模块和验证，不展开公式，不在开头堆叠背景。LC-SOFT CRSID 的摘要应围绕两个缺陷展开：hard SID 的刚性分配，以及长尾物品私有表示学习不足。

### 3.2 引言

建议采用以下段落顺序：

1. **任务背景**：说明序列推荐及语义表示的作用；
2. **已有进展**：从 ID-based 推荐过渡到内容增强和 Semantic ID；
3. **具体挑战**：分别解释 hard SID 边界误分配和长尾表示不稳定；
4. **方法洞见**：多槽重合可以提供离散局部一致性证据；
5. **方法概述**：依次介绍 Local-Consistent Soft SID、协同残差表示和可靠度校准；
6. **贡献列表**：通常使用三点，每一点对应一个机制和一组实验；
7. **章节安排**：用一段简短文字结束引言。

近期 WISE 论文经常明确列出两个挑战，再逐一映射到方法模块。不要只写“现有方法效果有限”，而要说明限制产生的技术原因。

### 3.3 方法章节

WISE 推荐论文的方法章节通常采用：

```text
3 Methodology
3.1 Problem Definition
3.2 Overall Framework / Hard SID Construction
3.3 Core Module I
3.4 Core Module II
3.5 Prediction and Objective Function
```

每个模块应遵循以下局部结构：

1. 先说明为什么需要该模块；
2. 用文字描述输入、处理过程和输出；
3. 再给出必要公式；
4. 紧接公式解释各符号及其直观含义；
5. 最后说明该设计解决了哪个前述挑战。

不应连续堆叠多个公式后再统一解释。非原创基础模块应简写并引用已有方法，将篇幅集中到核心创新。

LC-SOFT CRSID 建议使用：

```text
3 Methodology
3.1 Problem Definition and Overview
3.2 Hard Semantic ID Construction
3.3 Local-Consistent Soft SID Construction
    3.3.1 Multi-slot Local Semantic Neighborhood
    3.3.2 Slot-wise Candidate Token Estimation
    3.3.3 Hard-token Anchoring and Reliability
3.4 Collaborative Residual Item Representation
3.5 Frequency- and Reliability-aware Residual Allocation
3.6 Sequential Prediction and Optimization
```

### 3.4 实验章节

近期 WISE 推荐论文通常按照以下顺序展开实验：

```text
4 Experiments
4.1 Experimental Settings
    4.1.1 Datasets
    4.1.2 Baselines
    4.1.3 Evaluation Metrics / Protocol
    4.1.4 Implementation Details
4.2 Overall Performance
4.3 Ablation Study
4.4 Parameter or Further Analysis
```

结果分析倾向于列出两至四条观察，每条观察都需要引用表格中的具体趋势。消融实验必须与引言中的贡献点和方法模块一一对应。

## 4. 应用于 LC-SOFT CRSID 的写作规则

- Hard SID/RQ-KMeans 属于基础模块，只保留输入、分层量化结果和槽位偏移，不展开标准算法推导。
- Local-Consistent Soft SID 是主要创新，需要按照“动机 → 邻域 → 邻居投票 → 支持度过滤 → hard-token 锚定 → 可靠度”的顺序解释。
- 每个公式之前至少提供一句目的说明，公式之后立即解释变量及其作用。
- Collaborative Residual Item Representation 必须说明共享残差由 Soft SID 共享，而协同信号来自推荐损失。
- 频率和可靠度分配公式后需要解释高频、长尾、高可靠度和低可靠度四种情况。
- 方法概述图应显示离线构造和在线训练的边界。
- 正文中不保留代码路径、行号或实现备注；这些内容仅用于内部核对。
- 不声称首次提出 Soft SID，应将贡献限定为基于多槽重合邻域的逐槽局部分布构造。

## 5. 风格边界

应学习近期 WISE 论文的章节组织和表达节奏，但不复制其具体句子。部分 WISE 论文存在宽泛的效果表述和较弱的语言细节，本文应保留其紧凑结构，同时采用更严格的 claim–evidence 对齐：没有实验支持时，不使用“显著提升”“有效解决”或“优于现有方法”等结论性措辞。
