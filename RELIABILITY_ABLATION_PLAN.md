# Reliability-Aware Evidence Ablation Plan

本文档记录本轮在 EviQSD 上扩展的版本管理、实验目录和分析方式。所有版本都通过命令行参数控制，不覆盖当前最好模型。

## 1. Version List

| 版本 | 目录名 | 核心改动 | 目的 |
|---|---|---|---|
| SASRec | `exp_sasrec` | `sem_weight=0` | ID-only baseline |
| QSD-base | `exp_qsd_k8_sem010` | 原始语义分支 | 判断 semantic branch 基础收益 |
| Binary Evidence | `exp_evi_binary_f020_k8_sem010` | `history_overlap` | 当前 Office 最优规则版 |
| Strength Evidence | `exp_evi_strength_f020_r100_k8_sem010` | 频次 + 时间衰减 + 饱和证据 | 区分一次证据、多次证据、近期证据 |
| Strength + IDF | `exp_evi_strength_idf_f020_r100_k8_sem010` | evidence strength × token specificity | 抑制热门 semantic token |
| Cross + IDF | `exp_evi_cross_idf_f020_r100_c020_k8_sem010` | 同槽位证据 + 小权重跨槽位证据 | 测试跨槽位共享是否能补足 under-sharing |
| Learnable Reliability | `exp_evi_learnable_f020_r100_c020_k8_sem010` | MLP 估计 token reliability | 验证可学习证据估计是否优于规则 |
| Hub Penalty | `exp_evi_binary_hubpen005_f020_k8_sem010` | 对注意力关注 hub token 做惩罚 | 验证 semantic hubness 假设 |
| EC Fusion | `exp_evi_binary_ecfusion_f020_lfloor020_k8_sem010` | 根据 Evidence Coverage 动态降低语义分支权重 | 避免低证据样本被语义分支伤害 |
| Combined | `exp_evi_strength_idf_ecfusion_f020_r100_lfloor020_k8_sem010` | Strength + IDF + EC Fusion | 候选最终增强版 |

## 2. New Arguments

```text
--evidence-gate strength
--evidence-gate strength_idf
--evidence-gate cross_strength_idf
--evidence-gate learnable
--evidence-recency-weight FLOAT
--evidence-cross-weight FLOAT
--hub-penalty-weight FLOAT
--semantic-fusion fixed|evidence_coverage
--fusion-floor FLOAT
```

## 3. Core Formulas

### Evidence Strength

\[
c_{u,i,f}=\sum_{j\in S_u}\mathbb{I}(z_{j,f}=z_{i,f})\exp(-\lambda(1-p_j)),
\]

其中 \(p_j\in[0,1]\) 表示历史位置，越接近 1 越新。

\[
\hat{c}_{u,i,f}=1-\exp(-c_{u,i,f})
\]

\[
w_{u,i,f}=\rho+(1-\rho)\hat{c}_{u,i,f}
\]

### Token Specificity

代码中使用已归一化的 token hubness，specificity 定义为：

\[
spec_f(k)=1-hub_f(k)
\]

Strength + IDF 版本：

\[
R_{u,i,f}=\hat{c}_{u,i,f}\cdot spec_f(z_{i,f})
\]

\[
w_{u,i,f}=\rho+(1-\rho)R_{u,i,f}
\]

### Evidence Coverage Fusion

\[
EC(u,i)=\frac{1}{F}\sum_f\mathbb{I}[\exists j\in S_u,z_{j,f}=z_{i,f}]
\]

\[
\lambda_{u,i}=\lambda(\rho_{\lambda}+(1-\rho_{\lambda})EC(u,i))
\]

\[
s(u,i)=s_{id}(u,i)+\lambda_{u,i}s_{sem}(u,i)
\]

## 4. Run Command

Linux 服务器上运行：

```bash
chmod +x scripts/run_beauty_reliability_ablations.sh

PYTHON_BIN=python \
DATASET_DIR=runs/beauty \
SEMANTIC_IDS=runs/beauty/semantic_ids_rq.json \
OUT_ROOT=runs/beauty/reliability_ablation \
BATCH_SIZE=512 \
EVAL_BATCH_SIZE=2048 \
bash scripts/run_beauty_reliability_ablations.sh
```

如果显存足够，可以把 `BATCH_SIZE=1024`。如果全量评估 OOM，先把 `EVAL_BATCH_SIZE=1024`。

## 5. Analysis Protocol

跑完后先看：

```bash
python scripts/summarize_experiments.py \
  --root runs/beauty/reliability_ablation \
  --metric NDCG@10 \
  --top-k 50 \
  --csv runs/beauty/reliability_ablation/experiment_summary.csv
```

分析顺序：

1. `Binary Evidence` 是否超过 `QSD-base`：判断当前 evidence 思路在 Beauty 是否成立。
2. `Strength Evidence` 是否超过 `Binary Evidence`：判断频次和时间衰减是否有效。
3. `Strength + IDF` 是否超过 `Strength Evidence`：判断热门 token 抑制是否有效。
4. `Cross + IDF` 是否超过 `Strength + IDF`：判断跨槽位辅助证据是否真的补足 under-sharing。
5. `Learnable Reliability` 是否超过规则版：判断 MLP 是否学到了有效 reliability，而不是过拟合。
6. `EC Fusion` 是否在低 evidence 样本上更稳：需要后续分组评估验证。
7. `Combined` 是否超过所有单模块：如果没有，论文主方法应选择更简单、更稳定的单模块。

## 6. Expected Interpretation

如果 `Strength + IDF` 最好，论文主线应强调：

```text
User-specific evidence strength and global token specificity jointly determine semantic-token reliability.
```

如果 `EC Fusion` 最好，论文主线应强调：

```text
Semantic branch should contribute only when candidate semantics are supported by user historical evidence.
```

如果 `Binary Evidence` 仍最好，则说明当前数据规模下简单可解释规则比复杂 reliability 更稳，论文可以保留 Binary Evidence 作为主方法，把其他版本作为 negative/diagnostic ablation。

