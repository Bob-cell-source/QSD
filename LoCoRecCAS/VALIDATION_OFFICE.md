# LoCoRec-CAS：Office 初步验证（2026-09-09）

本轮实现与数值验证通过，但冻结骨干实验没有支持 context-adaptive utility supervision 优于静态共享或全局 gate。该结论只针对本轮配置，不等于完整联合训练或该研究方向已被否定。

## 实验范围

- 环境 CUDA 不可用，使用 CPU、2 个线程。
- 数据未抽样：训练 38,543 个 next-item 样本，valid/test 各 4,905 个样本，catalog 2,420 items。
- 从 `runs/office/locorec_loo_delta2_20260614/best.pt` 迁移可复用参数；按新稿公式重建全层 overlap 的完整候选并集（delta=3，H=50），不使用旧 top-M 或 LOO neighborhood。
- Static warm-up 3 epochs，lr=1e-4，g=1。三轮 valid NDCG@10 为 0.04933、0.06245、0.06698；选第三轮作为共同骨干。
- 冻结全部 item/sequence/assignment 参数，缓存相同 context 和负样本。三个控制器初始 gate 均为 0.9，gate lr=1e-3，最多 10 epochs，patience=4，lambda_util=1，T=0.1。
- 各自按验证 NDCG@10 选模型；test target 不参与训练或选优。Global 最佳 epoch=4，两个 context 版本最佳 epoch=1。
- 保留完整候选端 LayerNorm；全库评分使用保留动态方差的等价计算，未使用原式（30）的错误线性展开。

## 排名结果

| 方法 | Valid NDCG@10 | Test NDCG@10 | Test HR@10 |
|---|---:|---:|---:|
| Static：g=1 | 0.06698 | 0.05447 | 0.09786 |
| 无候选 shared residual：g=0 | 0.06094 | 0.04499 | 0.08563 |
| 固定 g=0.5 | 0.06619 | 0.05114 | 0.09480 |
| 学习全局 gate | 0.06711 | 0.05396 | 0.09704 |
| Context gate，仅 CE | 0.06679 | 0.05163 | 0.09276 |
| Context gate，CE + utility BCE | 0.06645 | 0.05188 | 0.09562 |
| 打乱 utility gate 的用户对应 | 0.06630 | 0.05176 | 0.09480 |
| 所有用户共用 utility gate 均值 | 0.06621 | 0.05146 | 0.09439 |

Utility 版本相对 static 的 Test NDCG@10 变化为 -4.76%，相对 learned global 为 -3.87%。
与普通 context gate 的微小差异、与打乱用户绑定的差异，其配对 bootstrap 区间均跨 0。
与 static/global 的配对差异在本 seed 下为负，95% bootstrap 区间分别约 [-0.00432, -0.00077] 和 [-0.00354, -0.00072]。这些是固定训练结果上的用户采样区间，不代表多 seed 重训结果。

## 效用是否被学到

- 最佳 utility controller 的测试 gate 均值约 [0.580, 0.592, 0.573, 0.558]，标准差约 [0.123, 0.116, 0.124, 0.126]。权重随 context 变化，但变化本身不能说明有效。
- 其预测 gate 与 held-out utility target 的逐层相关系数约 [0.036, 0.002, 0.028, 0.003]。
- Utility target MSE=0.09603；只预测训练集各层 target 均值的 MSE=0.07819，常数对照更好。
- 目标并非全部接近同一个值：测试 target 的逐层标准差约 0.27–0.29，只是平均值接近 0.5。

额外训练了一个仅使用 utility BCE 的线性探针（相同冻结 h，30 epochs；从训练 target 均值初始化，包含 epoch=0 的常数对照；按 valid BCE 选优）。最佳 epoch=0：训练得到的线性探针没有超过常数预测。该探针与正式推荐模型分开，不修改其 checkpoint。
重采样 held-out negatives 后，delta 的逐层相关性约 [0.950, 0.928, 0.943, 0.945]，正负号一致率约 88.2%–90.7%。因此不能把探针失败主要归因于本次负采样完全随机。

## 实现检查

- `OMP_NUM_THREADS=1 python -m pytest -q LoCoRecCAS/tests`：9 passed。
- 覆盖精确 prior、空邻居回退、表示公式、padding、LN 分数及 gate 梯度、utility detach、联合反向传播、冻结/联合小数据端到端流程和独立复评。
- 使用实际 Office static checkpoint 与 utility head 独立复评，所有测试指标与 results.json 完全一致。

## 结论边界和下一步

当前数据支持“在本轮冻结表示上，线性 context controller 未学到有用的逐层共享效用”。它不支持将本方法描述为已解决 negative transfer 或已实现有效 context-adaptive sharing。
本轮只进行了 3 轮静态 warm-up，其验证指标仍在上升；未验证 backbone 收敛、多个 seed、lambda/T 扫描或完整 Office 联合训练。最直接的后续验证是先让静态 warm-up 的验证表现稳定，再在相同骨干上重复这些对照；联合训练时仍必须保留 static/global/context-without-utility 对照。
代码提供 `--protocol joint`，该路径已通过小数据端到端测试，但不能把它记作本轮已跑完的完整 Office 实验。

## 产物

- `runs/office/locorec_cas/frozen_probe_seed2026/results.json`：完整对照指标与 bootstrap。
- 同目录 `warmup_history.json`、`gate_history.json`：逐 epoch 记录。
- 同目录 `utility_probe.json`：效用可预测性与负采样稳定性。
- 同目录 `static_best.pt`、`*_head.pt`：可直接复评的模型。
- `LoCoRecCAS/README.md`：实现定义、运行与复评说明。
