# 单共享表 LoCoRec：统一协议对照

**当前的新结构从头训练及其消融见 [CONSOLIDATED_VALIDATION.md](CONSOLIDATED_VALIDATION.md)。** 下述最初四模型实验用于诊断原 LoCoRec，不是 CAS 的消融，也不是新结构完整验证。

验证原 LoCoRec 能否简化为 `v_i = LN(A_i E + P_i)`，同时区分 Hard SID、Soft SID 与完整融合结构的收益。

## 四个模型

| variant | 表示和机制 |
|---|---|
| `id` | `LN(P_i)`；匹配的 ID-only 控制，不宣称逐项复现某篇论文的标准 SASRec |
| `hard` | `LN(mean_l E[z_i^l] + P_i)`；一张共享表和一张私有表 |
| `soft` | `LN(mean_l sum_k a_ilk E[k] + P_i)`；同一共享表，加入原局部候选与 prior-guided assignment |
| `full` | 原 `LoCoRec/locorec/model.py` 的三分支表示、频率/可靠性层次 gate、gate KL 和 private margin |

Soft 的 selector 用于学习 A，不是额外的表示分支。简化模型没有 semantic basis/shared residual 的双表，没有 alpha/gamma gate 或辅助正则。

所有版本都使用同一个原始 all-level SID-overlap scope 构造（delta=3、H=50、M=4、support=.05、hard anchor=1、squared prior；固定预处理 tie seed=2026）。Hard 只用固定 SID；Soft 与 Full 的候选和 prior 完全相同，不在此实验修改邻居排序或可靠性算法。

## 公平性与差异

- 全部从头训练，不加载旧 checkpoint。
- 同 seed 的 encoder、private embedding、共享表及适用的 selector 使用匹配初始化。
- 相同训练样本顺序与逐 epoch 负采样随机种子，100 个不重复负样本，屏蔽所有训练已知 positives。
- 相同 SASRec encoder：dim=128、max_len=50、2 heads、2 layers、dropout=.2。
- 所有版本关闭 item-side dropout，仅在 encoder 使用共同 dropout。Full 的结构来自原实现，但这一 dropout 控制与某些历史原版运行不同；不能把重跑数字当作旧论文结果的复现。
- 相同 sampled CE（未除 sqrt(d) 的 dot-product）、AdamW lr=.001、weight decay=.0001、clip=5。
- Full 保留原 gate 的 10-epoch warm-up、0.1 倍 gate lr、KL=.05、private penalty=.1；这是 Full 的特有机制，不能声称四个版本总 loss 完全相同。可设 `--gate-kl-weight 0 --private-weight 0` 在新目录另跑纯 CE 的 Full 对照。
- 相同最大 100 epochs、patience=10，统一在第 20 轮之后允许 early stopping，避免 Full 的 gate 尚未启动便终止。各模型按自己的 valid NDCG@10 选 checkpoint。
- Full-catalog 评估，屏蔽全部历史但保留当前 target。测试只用于最终报告。
- 所有模型均使用候选/历史 item LayerNorm。ID-only 是为隔离共享结构构造的匹配控制，不等价于无 item-LN 的其他 SASRec 实现。

## 运行

仓库根目录：

```bash
python -m LoCoRecSimple.experiment \
  --output-dir runs/office/locorec_simple_matched_20260910 \
  --device cuda
```

默认顺序运行 4 个模型 × seeds 2026/2027/2028。单个模型可用：

```bash
python -m LoCoRecSimple.experiment \
  --output-dir runs/office/simple_soft_probe \
  --variants soft --seeds 2026 \
  --device cuda
```

每个模型/seed 输出 best.pt、history.json、result.json 和逐用户 test_per_user.pt；根目录 results.json 逐个追加完成的运行，所有实验结束后 summary.json 给出 NDCG@10 mean/std 和参数数量。同一完整命令重启会跳过参数完全相同的已完成运行；未完成运行会从头重跑。不要在同一个 output-dir 混用不同参数或运行列表。

## 判断标准

1. Hard vs ID：确定单共享表本身的收益。
2. Soft vs Hard：检验局部 assignment 校准的独立价值。
3. Soft/Hard vs Full：检验多分支 gate 是否值得保留。
4. 多 seed 均值和逐 seed 差值优先于任意单次峰值；不以测试集选 seed。

Hard 得分高不应被人为压低。若 Hard 已与 Full 相当，简化本身就是重要发现；若 Soft 无稳定收益，应收窄 Soft SID 的贡献主张。

## Tests

```bash
OMP_NUM_THREADS=1 python -m pytest -q LoCoRecSimple/tests
```

覆盖表示公式、Hard/单候选 Soft 等价、padding、梯度、匹配初始化、完整历史屏蔽，以及四个版本的小数据端到端训练。

## 后续机制对照（2026-09-10）

- `full_fixed`：冻结原先验 gate 全程，取消辅助损失。
- `full_global`：alpha prior 统一为 catalog 均值，以两个全局参数代替 gate MLP；保留原 bounded correction、10 轮冻结和 gate 学习率。它不是任意范围的无约束全局融合。
- `full_equal`：有训练证据的物品使用 `[basis,shared,private]=[1,1,1]`，全程固定，取消辅助损失。上述控制保留原零训练频次物品的 private 屏蔽；因此“全局/等权”仅对有训练证据的物品严格成立。
- `hard_shuffled`：固定 seed=1701 随机置换非 padding SID 行，保留各层占用和完整 SID 组合的多重集合；从头训练，同训练 seed 对齐原 Hard。
- `id_transfer` / `id_transfer_shuffled`：只保留 ID-only 前向表示；固定 eta=.5，在梯度裁剪与 AdamW 前应用 `I+eta*(C-diag(C))`，其中 C 为各层 SID 分组均值投影的平均。后者采用相同 SID 行置换。此实验是机制探针，**不是 SEvo 的官方复现，也不主张优化器共享为新贡献**。

```bash
python -m LoCoRecSimple.experiment --output-dir runs/office/fusion_controls_20260910 --device cuda --variants full_fixed full_global full_equal --seeds 2026
python -m LoCoRecSimple.experiment --output-dir runs/office/sid_permutation_control_20260910 --device cuda --variants hard_shuffled --seeds 2026 2027 2028
python -m LoCoRecSimple.experiment --output-dir runs/office/sid_gradient_transfer_probe_20260910 --device cuda --variants id_transfer id_transfer_shuffled --seeds 2026
```

汇总可通过 `--reference-dir` 引入原四模型结果，对共同种子计算配对差值；汇总会检查共同训练协议字段并拒绝重复 model/seed：

```bash
python -m LoCoRecSimple.summarize --run-dir runs/office/fusion_controls_20260910 --reference-dir runs/office/locorec_simple_matched_20260910
```

`id_scaled / hard_scaled / soft_scaled / full_scaled` 是统一除以 `sqrt(dim)` 后的训练对照，其他设置不变。全库排序对这一正的常数不变，因此评估实现仍可使用未缩放 logits。原始未缩放结果不可用于证明结构在充分调参后的优劣；见 VALIDATION_OFFICE.md 的初始化审计。
