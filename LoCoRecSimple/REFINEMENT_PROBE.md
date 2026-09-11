# 下游共享参数细化：最小机制实验

状态：原型，不主张新颖性已确认，也不是 SteepGS 或 Splitting Steepest Descent 的直接复现。

## 模型和范围

`v_i = LN(P_i + mean_l T[lookup(i,l)])`，SASRec，训练 CE logits 除以 sqrt(d)。冻结原始 SID 整数表，保持 tokenizer/codebook 不变；仅调整下游 lookup 表，并训练下游共享表与 private 表。完全关闭 Soft SID、统计先验 gate、辅助损失。存在 private 表，因此不声称分裂扩大了最终物品表示的可表达集合，研究的是协同参数绑定和训练归纳偏置。

从随机参数开始共同 warm-up 3 轮。随后复制同一模型和 AdamW 状态，运行五个 continuation：

| 版本 | 选择父组 | 组内二分 |
|---|---|---|
| no_split | 无 | 不分裂 |
| gradient_split | 梯度二次代理收益最高 | 中心化物品梯度的第一主成分，按中位数平衡二分 |
| same_parent_random | 与 gradient_split 相同 | 打乱成员，保持每个子组大小相同 |
| random_split | 随机父组 | 随机平衡二分 |
| frequency_split | 父组训练物品频次总和最高 | 随机平衡二分 |

每个父组至少8个物品，最多选择32个父组，一次分裂。四个分裂版本增加相同数量的128维共享行；禁止以测试指标决定预算或子组。随机子组固定实验种子。父组仍包含多个物品，不直接变成额外 item-private embedding。

## 选择信号

对固定 warm-up 模型，eval 模式关闭 dropout。训练用户按固定种子1707随机分为两组，每组仅使用其训练前缀，分别最多采样64个 batch。收集 LayerNorm 前物品表示的梯度，包括历史与候选两条路径，除以 SID 层数以还原每层 shared 参数的贡献，再按 probe 样本数归一化。A/B 两组的训练用户不同，但模型 warm-up 见过两组：B 是独立梯度估计审计，不是未见用户泛化评估。

只使用 A 建立所有拟学习分区。对一个父组 C=C0∪C1，设 G_C=sum_i g_i，使用 group-size 加权二次代理：

`min_delta G_C^T delta + |C| ||delta||^2 / (2 eta)`。

解除绑定相对于保持绑定的代理最优收益（去除共同常数 eta/2）：

`||G_C0||^2/|C0| + ||G_C1||^2/|C1| - ||G_C||^2/|C|`。

这是非负的拟合分数，**不等于已证明的负迁移或真实测试收益**。平衡 PCA 二分只是搜索该代理的一个启发式，不保证最佳二分。

审计 B：固定 A 的分区和拟更新方向，报告

`G_A,C0 dot G_B,C0 / |C0| + G_A,C1 dot G_B,C1 / |C1| - G_A,C dot G_B,C / |C|`。

该值可为负，反映 A 中的解除绑定方向能否在 B 中复现。B 不参与分裂选择。以上只分析局部一阶项和选定二次代价，不提供 AdamW、LayerNorm 后非线性或验证损失下降保证。训练梯度也包含负采样噪声和物品角色差异，不能将全部方差解释成有害冲突。

## 分裂与公平性

每个子组选择父行或新增克隆行；初始两行向量完全相同，Adam 的一二阶矩同样复制。因此分裂瞬间表示和预测保持一致。模型仍然训练同一 SASRec、shared/private 和 LN。比较在完全相同的共同 warm-up 后开始，各版本独立用验证集选择 continuation checkpoint（包含分裂初始点）。这不是五次独立从头训练，也不引用任何历史运行作为数值对照。

默认单 seed2026、Office、最多总30轮、patience8、至少10轮 continuation；其他设置 dim128、maxlen50、2层2头、dropout.2、batch256、100 negatives、AdamW lr.001/wd.0001、clip5、全库评估。此轮为机制探针，不是充分调参后的论文比较。只有信号复现和相同预算控制支持假设后，才值得扩大种子与数据集。

```bash
python -m LoCoRecSimple.refinement_experiment \
  --output-dir runs/office/refinement_probe_20260910 \
  --seeds 2026 --device cuda
```

输出共同 warmup.pt、梯度 A/B、分区与审计分数、五个版本的 history、best.pt、result.json 和逐用户 NDCG。主目录 summary.json 在所有版本结束时生成。由于此目录先按 seed 分组，不应使用原 summarize.py 混合统计；使用本模块自己的 summarize。

## 来源与边界

- Splitting Steepest Descent: https://arxiv.org/abs/1910.02366
- SteepGS: https://vita-group.github.io/SteepGS/
- 已有推荐 embedding 分配：AdaEmbed https://www.usenix.org/conference/osdi23/presentation/lai ，CAFE https://arxiv.org/abs/2312.03256 。不能将“按梯度分配 embedding 容量”称为首次提出。
- 当前原型需要检验的窄问题是：冻结 SID 下，组内训练方向驱动的平衡参数解除绑定，能否优于相同预算的无信息分裂。

相关知识：固定 SID 不等于冻结下游 token embedding。TIGER 两阶段训练 https://arxiv.org/html/2305.05065v2 ，VQ-Rec 可训练 code 表 https://github.com/RUCAIBox/VQ-Rec/blob/master/vqrec.py ，ETEGRec 联合 tokenizer/recommender 学习 https://arxiv.org/abs/2409.05546 。下游 embedding 接受推荐监督并非本方法贡献。
