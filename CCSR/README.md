# Context-Conditioned Sharing Resolution：最小 Hard SID 版本

实现 Hard SID + 多分辨率共享表示 + private identity + context stopping。复用 `LoCoRec.locorec` 的 SASRec、序列数据集和 SID 编号规则。不构造 local candidates，不使用 Soft SID、attention selector、alpha/gamma gate、频率先验、gate KL 或 private regularizer。优化器为 Adam，weight decay 为 0。

## 实现对应

- `model.py`：离线 prefix group/entropy/cost、item representation、线性 stop head、survival distribution、expected CE、Hard/Soft scoring、Hard checkpoint 迁移。
- `trainer.py`：两阶段训练、lambda ramp、全库分块评估、每档固定评分与路由诊断、可选 Mem/Gen 分组。
- `evaluate.py`：独立加载新 checkpoint 评估，无需重新训练。
- `tests/test_core.py`：公式、梯度、padding、路由、评估、checkpoint 与端到端测试。

每层 `e_l = basis_projection(E_b[z_l]) + E_s[z_l]`，前 r 层均值为 `g_r`；`v_r = LN(g_r + rho_r * P_i)`，最后一档为 `LN(g_L + P_i)`。projection 保留原 LoCoRec 的 bias 以兼容 checkpoint；同一个线性投影作用于每层后取平均，与先平均再投影等价。各层 token 使用互不重叠的编号空间。

历史仅计算最后一档表示。候选得分统一除以 `sqrt(d)`；逐档计算 sampled CE，再求 `sum(pi * CE_r)`，不先混合表示或 logits 后计算 CE。唯一附加项是 `lambda_res * sum(pi * rho)`。

stop weight 初始化为 0，bias 初始化为 `logit(1 / [L+1, L, ..., 2])`。默认前 3 个 epoch 固定均匀 pi 并冻结 stop head；之后 lambda 系数依次为目标值的 `0, .25, .5, .75, 1`。只有第二阶段的 checkpoint 可成为 `best.pt`；选择指标为验证集 **Hard NDCG@10**，不会用测试集选模型。

## Office 启动

在仓库根目录运行（命令为正式训练，CPU 可改 `--device cpu`）：

```bash
python -m CCSR.train \
  --dataset-dir runs/office \
  --semantic-ids runs/office/semantic_ids_rq.json \
  --init-checkpoint runs/office/locorec_loo_delta2_hard_m1_20260614/best.pt \
  --output-dir runs/office/ccsr/lambda_0.03 \
  --device cuda \
  --lambda-res 0.03 \
  --warmup-epochs 3 \
  --lambda-warmup-epochs 5
```

默认 dim=128、max_len=50、heads=2、layers=2、dropout=.2、100 个不重复负样本、batch=256、lr=.001。其他 checkpoint 的模型结构参数需显式设为源 checkpoint 的值。SID 必须覆盖全部 item、各层 code 必须落在对应 codebook 范围；padding item 为 0。

初始化必须二选一：`--init-checkpoint PATH` 或显式 `--from-scratch`。后者仅用于随机初始化对照和 smoke test，不等同于建议的 Stage 0。

迁移检查 SID 表完全一致且每层只有一个有效 token，拒绝 Soft SID checkpoint；复用 encoder、semantic basis、basis projection、shared/private residual、item LayerNorm，不加载旧 gate/selector/先验。缺少 shared residual 的 `HardSIDFusion` checkpoint 不兼容，会明确报错。应使用上面的 `hard_m1` 消融 checkpoint。

扫描建议的 5 个 lambda：

```bash
bash CCSR/run_office_sweep.sh
```

脚本顺序运行实验，输出 `runs/office/ccsr_sweep/selection.json`，按验证集 Hard NDCG@10 选择 lambda。`DEVICE=cpu EPOCHS=8 bash CCSR/run_office_sweep.sh` 可覆盖设备和训练轮数。

## 固定分辨率基线

每个 adaptive checkpoint 自动评估 always R1 ... always ID。这些是**同一组多分辨率参数的固定评分对照**。

要分别优化固定分辨率模型，在上述训练命令中添加 `--fixed-resolution 1`（或 2...L+1），并使用独立 output-dir。此时 pi 固定 one-hot，不训练 router，不用 uniform warm-up 或 resolution cost 惩罚；历史仍使用 ID 表示。论文中比较独立训练的 fixed baselines 时，应为各 baseline 公平选择验证集配置并运行多个 seed。

## 输出与推理

- `resolution_preprocess.json`：H、rho、各深度 group 数和最后两档是否相同；prefix group 表同时保存在 checkpoint buffer 中。
- `initialization.json`：迁移参数列表或显式 scratch 标记。
- `history.json`：逐 epoch loss、各档 CE、实际 lambda、训练/验证 router 统计、所有模式验证指标。
- `best.pt`：第二阶段验证集 Hard NDCG@10 最好的模型及配置。
- `test_metrics.json`：best epoch、对应验证指标、所有模式测试结果。

`ranking.hard` 为 `argmax(pi)` 选择档位的主结果；`ranking.soft` 为各档 logits 的 pi 加权和；`ranking.1` ... `ranking.(L+1)` 为固定档位。每个模式提供 HR/Recall/NDCG@5,10,20。

`router.probability_distribution` 是平均 pi，`selected_distribution` 是 argmax 实际选择比例；同时记录 expected resolution、cost、entropy。前者不能当作 Hard 实际使用比例，uniform 初始化时 argmax 平局会选第一档。

全库表示每次评估预计算并缓存于 CPU，按块送入设备；Hard scoring 先按用户所选档位分组再做矩阵乘法。评估保留各模式流式 top-k，不保存整个 B×N×R 得分张量。屏蔽全部已见历史，但保留本次 target（允许重复消费）。

```bash
python -m CCSR.evaluate \
  --checkpoint runs/office/ccsr/lambda_0.03/best.pt \
  --split test --device cuda \
  --output runs/office/ccsr/lambda_0.03/reeval.json
```

该入口同时输出 Hard、Soft 和全部固定档位。

## 可选 Mem/Gen 分析

`--group-labels labels.json` 接受 `{"valid": ["Mem", "Gen", ...], "test": [...]}`。数组与 `NextItemDataset` 的评估样本顺序对齐：按 sequences.json 原顺序，跳过长度不足 3 的行，每个剩余行一个样本。分组由已有 Meta 划分提供，不自动用频率替代 Mem/Gen 定义。输出各组 pi/Hard 分布、E[R]、E[C] 和路由熵；没有标签时不生成 Mem/Gen 结论。

## 数学边界与实验解释

如果所有 item 的第一层 SID 都相同，rho_1=0，该档没有 private anchor；因此一般正确范围是 `0 <= rho_1 <= ... <= 1`。不加 epsilon 修改原公式。

如果完整 SID 已唯一标识全部 item，rho_L=1，此时 R_L 与 ID 表示相同。即使 rho<1，private embedding 的可学习幅度也意味着 rho 只是系数与 specificity cost，并非信息容量或推理计算量的严格上界。

lambda=0 不保证最深档最优；Soft 优于 Hard 也不能单凭这一点认定 router 不够 sharp。是否成立需看验证选择后的正式测试、多 seed、独立 fixed baselines，不能由实现正确或 smoke test 推出。

## 验证

```bash
OMP_NUM_THREADS=1 python -m pytest -q CCSR/tests
```
