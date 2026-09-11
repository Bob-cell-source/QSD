# LoCoRec-CAS validation

实现 Local Sharing Scope + Context-Adaptive Shared Residual + detached LOO utility supervision。入口支持完整两阶段联合训练，以及本轮首先验证的冻结骨干对照。

## Method 对应与修正

- 固定 Hard SID；邻居按全部层 overlap >= delta 取 Top-H（排除自己，确定性打破平局）。默认 delta=3、H=50。
- 候选为 Hard token 与邻居 token 的完整并集，不套用旧实现的 top-M 或 minimum support 截断。
- prior 严格使用 `(local_support + hard_anchor)^2` 后归一化；空邻居退化为 Hard-only。
- query 明确定义为 prior 加权的 assignment embedding，再经过 Wq；a 只依赖 item，与用户 context 无关。
- `base_i = b_i + p_i`，`shared_i[l] = s_i^(l)`。
- 历史始终为 `LN(base_i + mean_l shared_i[l])`。
- 候选为 `LN(base_i + sum_l g[u,l] shared_i[l] / L)`。
- 每个训练样本的 all-sharing 和各个 LOO 使用同一个 h、同一组候选、相同已计算的表示。只移除候选 shared residual，不重新编码历史，也不将 1/L 改为 1/(L-1)。
- `delta = CE_without_level - CE_all`；`target = sigmoid(delta/T)` 全部 detach。用 BCEWithLogits 监督线性 context head。
- 目标只有 `CE_gated + lambda_util * BCE`。没有旧 alpha/gamma、频率先验、private penalty、gate KL、weight decay。
- gate 初始化为常数 .9（weight=0，bias=logit(.9)）；Stage I 精确固定 gate=1。Stage II 不人为修改 utility target 的中心。

**式（30）的线性 score 展开没有使用。** 候选端 LN 的方差取决于 context 与 item，不能丢弃。全库按候选块处理。

冻结骨干评估采用与 LN 完全等价的计算：令 `t_i=[base_i, s_i^1/L,...,s_i^L/L]`，各向量逐个减去自身通道均值得到 `phi_i`，`c_u=[1,g_u]`，`G_i=phi_i phi_i^T/d`。则

```
score(u,i) = ((h_u * gamma)^T sum_k c_uk phi_ik
              / sqrt(c_u^T G_i c_u + eps) + h_u^T beta) / sqrt(d)
```

该实现保留每个 user-item 的动态归一化分母。`h^T beta` 对候选为常数，计算 CE/排名时可省略。单元测试同时核对原始 LN 的分数与 gate 梯度。联合训练的 forward 直接执行 LN。

## 冻结骨干验证（本轮实验）

```bash
python -m LoCoRecCAS.experiment \
  --init-checkpoint runs/office/locorec_loo_delta2_20260614/best.pt \
  --output-dir runs/office/locorec_cas/frozen_probe_seed2026 \
  --device cpu --threads 2 \
  --warmup-epochs 3 --epochs 10 \
  --lr 0.0001 --gate-lr 0.001 \
  --lambda-util 1 --temperature 0.1
```

该入口：

1. 迁移 LoCoRec 的 encoder、semantic/shared/private embedding、selector、projection、LN；忽略旧 gate 和 scope buffer。使用当前公式重建 scope。
2. 完整训练集上做 3 轮静态 warm-up，按验证集 static NDCG@10 保存 `static_best.pt`。
3. 从相同 static checkpoint 冻结 scope、item 表示与 sequence encoder，以 eval 模式缓存 context，不使用 dropout。
4. 分别学习 global gate、context gate（仅 CE）、context gate（CE+BCE）。三者共用训练样本、固定的 100 个不重复负样本和 epoch 打乱种子；各自按验证 NDCG@10 选 checkpoint。
5. 全库验证/测试，报告 static(g=1)、none(g=0)、half(g=.5)、global、context_rec、context_utility、utility_shuffled、utility_mean。只对 context_utility 的用户绑定做一次固定种子的 shuffle；mean 是该 split 所有 gate 的均值，与独立训练的 global 对照分开报告。
6. 在 valid/test 上用固定 sampled candidates 计算 LOO targets，仅作事后诊断，未参与 gate 训练或模型选优。报告 target MSE 与训练集 target 均值常数预测的 MSE、逐层相关性、delta 分布。

训练样本使用原 LoCoRec 的所有前缀（Office 38,543）；valid/test 各 4,905 个样本，catalog 2,420 items。全库评估屏蔽历史但保留本次 target。该验证不是 100-negative sampled ranking。

`results.json` 是最终对照结果；`warmup_history.json` 和 `gate_history.json` 是逐轮记录；`*_head.pt` 保存各自最佳 controller，`static_best.pt` 是共同骨干。配对 bootstrap 区间仅刻画此 seed 下测试样本差异，不代替多 seed 重训。

可以通过 `--static-checkpoint PATH` 复用静态 warm-up，改变 lambda/T/gate-lr 做新的验证配置。结构参数及 scope 参数必须一致。每次使用不同 output-dir；已有 results.json 的目录会拒绝覆盖。

## 完整 Stage II 联合训练

```bash
python -m LoCoRecCAS.experiment \
  --static-checkpoint runs/office/locorec_cas/frozen_probe_seed2026/static_best.pt \
  --output-dir runs/office/locorec_cas/joint_seed2026 \
  --device cuda --protocol joint --joint-mode context \
  --epochs 30 --lr 0.0001 --lambda-util 1 --temperature 0.1
```

联合训练模式每轮重新采样 negatives，所有目标从当前模型产生并 detach。`--lambda-util 0` 是无 utility supervision 的 context 对照；`--joint-mode global` 是全局 gate；`--joint-mode static` 是固定全开共享。联合实验需各自使用不同 output-dir。默认 protocol 是冻结骨干，不能将冻结实验结果写作联合训练结果。

## 已知解释边界

- `g=0` 仅关闭候选 shared collaborative residual，仍保留 SID semantic basis；不是纯 ID 模型。
- 冻结 tokenizer 不意味着 semantic basis 固定；本模型继承的 basis 与 shared 表仍为可训练表示，未增加解耦约束。
- LOO 标签是 all-sharing reference 下的训练预测效用，不是因果泛化收益；对多层同时关闭的效用不具有可加性保证。
- sigmoid target 在 delta=0 时为 .5，因此必须保留 half/global 对照，不能仅凭 CE+BCE 比 g=1 好就声称学到了 context。
- 从现有按验证集选优的 LoCoRec checkpoint 迁移，继承了既有的预训练/选优过程；各控制使用相同起点，不把这轮视为独立多数据集结论。

## Tests

```bash
OMP_NUM_THREADS=1 python -m pytest -q LoCoRecCAS/tests
```

覆盖候选 prior、空邻居、表示公式、padding、精确 LN 分数及梯度、detached utility、联合训练梯度、context 与候选独立性、冻结和联合训练的端到端流程。

## 独立复评

```bash
python -m LoCoRecCAS.evaluate \
  --checkpoint runs/office/locorec_cas/frozen_probe_seed2026/static_best.pt \
  --head runs/office/locorec_cas/frozen_probe_seed2026/context_utility_head.pt \
  --device cpu --split test \
  --output runs/office/locorec_cas/frozen_probe_seed2026/reeval.json
```

复评会校验 frozen head 对应的骨干。联合训练用 `--checkpoint .../joint_best.pt`，不传 `--head`。

## 本轮结果与效用探针

见 [VALIDATION_OFFICE.md](VALIDATION_OFFICE.md)。独立效用探针可以复现：

```bash
python -m LoCoRecCAS.probe_utility \
  --checkpoint runs/office/locorec_cas/frozen_probe_seed2026/static_best.pt \
  --output runs/office/locorec_cas/frozen_probe_seed2026/utility_probe.json
```

它只训练一个诊断用线性效用预测器，按验证 BCE 选优，并包含常数初始化（epoch=0）作为候选。另用第二组测试负样本检查 LOO target 稳定性；不修改正式推荐模型。
