# A/B/C 诊断复现

在项目根目录执行：

```bash
python -m GradientCalibratedSID.diagnose_abc \
  --source runs/office/gcss_20260910 \
  --output runs/office/gcss_diagnostic_abc_20260910 \
  --seed 2026 --device cuda
python -m GradientCalibratedSID.plot_abc \
  --root runs/office/gcss_diagnostic_abc_20260910
```

必须先完成 GCSS 的纯 Hard 训练和 positive-only profiling。诊断检查 signature checkpoint SHA256；不加载旧的 Hard+private 结果。固定训练设置来自该 checkpoint。首次运行前指定新的 output；相同 output 支持跳过已经完成的训练，但不要用它混合不同 source/seed。

## A

逐层统计共享 token 内全部无序物品对，比较既有 semantic neighborhood 中该层 token 不同的物品对，以及从全体有效物品均匀抽取的随机 pair。语义邻居去重，随机对排除 self、允许跨次重复，数量匹配 same-token pair 数。主表要求物品至少1次 profiling 观测，另做至少5次观测的敏感性分析。pair 之间不独立，不把 pair 数当独立统计样本计算显著性。

## B

严格计算 `C=(||sum g||²-sum ||g||²)/(n(n-1))`，包含无观测物品的零签名；同时报告实际有签名的物品数。每个 token 至少两个有签名物品才纳入表格。主分析至少10个测试实例，按 SID level 独立分四分位，报告 full-catalog Recall@10（单正样本下等于 HR@10）、NDCG@10、filtered full-catalog 正样本 CE，以及 C 与占用量/训练频率的相关性。测试时保留目标并过滤历史物品。

Atomic-ID 从随机初始化重新训练：一个物品 embedding table + item LN + 相同 Transformer、负采样、训练上限与 validation 选择。它与 SID 的参数量不同，会明确报告。`token_groups.csv` 保存 Hard−Atomic NDCG；`B_summary.json` 给出其与 C 的相关性。

## C

每层从至少8个物品且至少8个物品有3次 profiling 观测的 token 中，选择最多4个 C 最低的组；高 alignment 对照从 C 上半区选择，占用量尽量接近对应低 C 组。这里只按训练梯度选组，不根据测试表现选组。

低 C 组称为相对高冲突组，不预设其平均 C 为负。梯度分组使用原始未再次归一化的 g 做确定性二均值；随机对照在同一 token 内打乱分组标签，保持两个子组大小一致。16个 token 分裂只增加16行 embedding，两个对照臂参数量一致。原 token 行复制到新行，不加噪声；严格检查全部物品初始向量完全一致，单元测试还检查初始序列打分一致。

五组 continuation：no split、low-C guided/random、high-C guided/random。从同一收敛 Hard checkpoint 出发，重置 optimizer，配对 epoch 采样和 dropout RNG，保持相同 validation 早停设置，并允许 epoch0 被选为最佳。

`C_comparisons.json` 报告整体、受影响 target、未受影响 target 上的配对 NDCG 差及1000次用户 bootstrap 95%区间。受影响指 target 使用了任一被拆 token；未受影响 target 的历史仍可能包含受影响物品，因此这不是严格隔离的处理/未处理人群。训练 seed 的不确定性不能由用户 bootstrap 替代。

图脚本另存 `C_partition_geometry.json`，检查 guided 是否真的比随机拆分降低子组内部冲突。单 seed、收敛 checkpoint 上的局部干预，无论正负，都不能代表所有训练阶段或数据集。

## 测试

```bash
python -m pytest GradientCalibratedSID/tests/test_diagnose_abc.py -q
```

检查 pair 符号、C 的零签名分母、梯度聚类、随机 child size、参数量及拆分前后初始预测一致性。
