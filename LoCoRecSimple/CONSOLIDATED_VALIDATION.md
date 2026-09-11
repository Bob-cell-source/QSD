# 单共享表固定融合：从头训练验证

本轮专门验证简化结构，不将原 LoCoRec 的多模块对照称为新方法消融，也不将训练后 checkpoint 合并当作从头训练结果。

## 待验证模型

`v_i = LN(A_i T + beta_i Q_i + b)`。A 是原局部 Soft SID assignment；T 为单共享表；Q 为私有表；beta 是固定的物品级系数；b 为公共偏置。历史和候选采用同一静态表示。没有上下文 router、可学习融合 gate、双共享表、投影或辅助损失。

这只是待验证的简洁参数化，不宣称代数重写本身具有论文创新性。

## 初始化和训练边界

先创建随机初始化的原 Full，然后在**任何训练发生之前**用 compile_model 转换参数，使简化主模型与随机 Full 初始函数一致。此后只优化简化模型自身的参数，训练轨迹不等价于 Full。不加载任何训练 checkpoint。

这一初始化保留了原随机 Full 的公共偏置、固定融合先验及相关性；它不是各表独立 Xavier 初始化。若结果有效，还需研究初始化依赖，不能直接归功于结构。

## 本轮五个版本

| 版本 | 唯一变更或角色 |
|---|---|
| consolidated_scaled | 简化主模型，CE only |
| consolidated_hard_scaled | 在主模型上换成单候选 Hard SID，冻结不再起作用的 selector |
| consolidated_no_bias_scaled | 在主模型上将公共偏置清零并冻结 |
| consolidated_uniform_scaled | 在主模型上将正 private 系数替换为其均值；保留冷物品零系数 |
| full_scaled | 重新训练的原 LoCoRec 外部对照，保留其辅助损失 |

三个消融保留主模型其余初始张量。Hard 消融的私有表仍采用同一转换得到的初始值，以避免同时更换初始化；这检验训练中 Soft assignment 的价值，不是完全去除 Soft 信息的 tokenizer 独立基线。

## 预先固定协议

Office；seeds 2026/2027/2028；五个版本全部本轮重跑，不引入旧结果。dim=128、2 层、2 heads、maxlen=50、encoder dropout=.2、100 negatives、batch=256、AdamW lr=.001、wd=.0001、clip=5；CE logits 统一除以 sqrt(dim)。最大100轮，patience=10，第20轮起允许早停；各版本按各自验证 NDCG@10 选 checkpoint。测试采用全库排名，不以测试分数决定 seed 或 checkpoint。

主要比较：主模型减三个消融的同 seed 差值，以及主模型减 Full 的同 seed 差值。报告所有三个种子均值和样本标准差，不从中挑一个最好种子。此轮是固定协议机制实验，不是各方法充分调参后的最终论文比较。

```bash
python -m LoCoRecSimple.experiment \
  --output-dir runs/office/consolidated_ablation_20260910 \
  --variants consolidated_scaled consolidated_hard_scaled consolidated_no_bias_scaled consolidated_uniform_scaled full_scaled \
  --seeds 2026 2027 2028 --device cuda
```

原始输出位于上述独立目录。所有版本结束后：

```bash
python -m LoCoRecSimple.summarize --run-dir runs/office/consolidated_ablation_20260910
```

代码检查：14 项测试通过，包括初始函数等价、Hard 消融公式、冷物品屏蔽、梯度和四个新版本的端到端训练。数值测试不替代真实数据集效果。
