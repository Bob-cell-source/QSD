# Gradient-Calibrated Semantic Sharing

实现固定的最小方案：语义候选 + positive-target gradient calibration + weighted SID lookup。
仅有 SID embedding、一次 item LayerNorm 和原有 causal Transformer；没有 private ID、gate、额外 attention、辅助损失或动态刷新。

## 一条指令运行全部阶段

在仓库根目录执行：

```bash
python -m GradientCalibratedSID.run \
  --dataset-dir runs/office \
  --semantic-ids runs/office/semantic_ids_rq.json \
  --output-dir runs/office/gcss_20260910 \
  --seeds 2026 --device cuda --H 50 --M 4 --lambda-grad 1
```

默认 d=128、max_len=50、2 层/2 heads、dropout=0.2、100 negatives、batch=256、AdamW lr=0.001、weight_decay=0.0001、最多100 epochs、validation NDCG@10 patience=10。权重衰减沿用训练器的常规优化设置，不引入额外训练 loss；可用 `--weight-decay 0` 关闭。所有版本只优化 sampled CE，评估 full catalog，过滤历史交互。

多 seed 使用新目录并设置 `--seeds 2026 2027 2028`。相同目录支持跳过已经完成的阶段；改变数据、seed、H/M/lambda 或训练配置必须使用新目录。未完成的训练阶段从该阶段起点重跑，不恢复中断时的 optimizer 状态。每个 epoch 写 `history.json`，每个完成版本写 `result.json`，汇总见 `REPORT.md`。

## 数据与阶段

沿用现有冻结 tokenizer 的产物，不重复训练 tokenizer。数据目录需要 `sequences.json`、`stats.json`、`item_text_embeddings.npy`、`embedding_item_ids.json`；SID JSON 包含 `semantic_ids` 和 `codebook_sizes`。真实 item ID 必须为 1…N，0 为 padding；文本按 item ID 对齐。输入文本应为 tokenizer 前的冻结语义向量。源文件 SHA256 写入 `protocol.json`。

以下入口等价于统一命令的 `--stage`，其余参数需要保持一致：

| 入口 | `--stage` | 产物 |
|---|---|---|
| `02_build_semantic_candidates.py` | candidates | `semantic_candidates.pt` |
| `03_train_hard_sid.py` | hard | `seed*/stage1/hard_sid_checkpoint.pt` |
| `04_profile_item_gradients.py` | profile | `seed*/grad_signature.pt` |
| `05_build_gradient_calibration.py` | calibrate | `seed*/calibration.pt` |
| `06_train_calibrated_sid.py` | stage2 | `seed*/stage2/*/best.pt` |

例如：`python GradientCalibratedSID/04_profile_item_gradients.py`。它会先做32个物品的 debug，通过后才提取全部训练正样本的签名。也可以分阶段执行 `python -m GradientCalibratedSID.run --stage profile`。

现有 LoCoRec Hard+private checkpoint 与本方法不同，加载器明确拒绝；Stage 1 从随机初始化重新训练纯 Hard SID。Stage 2 的四个版本从同一个新 Hard checkpoint 初始化，重置 optimizer，并使用相同 epoch 级采样、负样本和 dropout 随机种子。最终 checkpoint 仅按 validation 选择，test 不参与选择。

## 数学实现约定

- 邻居排除自身，只要求 aligned SID overlap > 0。posting lists 避免分配 N×N 矩阵；先按 overlap、再按冻结文本 cosine 降序。仅两者完全相同时使用 item ID 确保确定性。极端高占用下，访问 posting lists 的总时间仍可能为二次量级。
- 保存原始 augmented count support `q=count/(|N_i|+1)`；Top-M 截断后不修改 q。原 Hard token 永远占第一个 slot；其余 support 相同时按最高排名邻居先后选取。
- 磁盘物品张量为 `[N,L,M]`、`[N,H]`、`[N,d]`，不含 padding item 行。候选 token 是按 level 偏移的全局 embedding ID，与分层 embedding table 等价；neighbor IDs 为原始 1-based item ID，0 padding。
- profiling 冻结全部模型参数并关闭 dropout。history 和 negative representations 无梯度；只有每行 positive pre-LN leaf 可求导。CE 使用 sum，使批量梯度与逐样本梯度一致。
- profiling 临时关闭 matmul/cuDNN TF32，避免不同 batch size 的卷积数值路径影响严格梯度检查；离开 profiling 后恢复原设置，训练配置保持不变。
- 每个样本先除以 `norm+eps`，再按 positive item 求均值。零梯度不计入有效次数，无观测物品保持零向量。**不再归一化 g_i，也不归一化候选 group prototype**。非有限梯度直接报错。
- compatibility 仅在邻居中按原始 SID token 分组，排除自身。零签名邻居仍参与均值分母，空组为0。
- `a=softmax(log(q)+lambda*c)` 仅作用于有效候选，padding 权重为0。固定权重为 buffer，不参与优化；训练前后检查 buffer SHA256。
- item 表示严格为 `LN(mean_l(sum_k a*E_l[k]))`。Stage 2 的历史和候选都使用这一 lookup。除原模型 backbone 外没有新模块。

## 四组对照

| 版本 | 唯一差异 |
|---|---|
| hard | M=1，继续训练原 Hard lookup |
| semantic_soft | 语义候选，lambda=0 |
| gradient_calibrated | 同候选，lambda>0 |
| random_compatibility | 每个 item/level 内随机置换有效 c，再用相同 lambda |

Random 保留每行兼容度数值的多重集合及 mask；只有一个候选时自然不变。四组没有额外可训练参数。Stage 1 原始 Hard 结果单独保存，不冒充 Stage 2 同训练预算对照。核心检验是 calibrated 相对 semantic_soft/random 的同 seed 差值，单次运行不能证明统计显著性。

## Profiling 验证

```bash
python -m pytest GradientCalibratedSID/tests/test_gcss.py -q
```

测试覆盖候选排序与支持度、M=1 等价、正样本与其他行负样本重复时的梯度隔离、逐样本/批量一致性、各 level 梯度的 1/L 关系、方向抵消、self exclusion、prototype 不归一化、lambda=0、随机对照、旧 checkpoint 拒绝，以及 CPU 上完整四组训练与重载。

真实数据诊断见 `profile_debug.json`、`profile_diagnostics.json` 和 `calibration_diagnostics.json`，包括冻结状态校验、逐样本误差、有效观测次数、签名模长分布、负兼容度比例与实际权重变化。debug 失败会在 Stage 2 前停止。
