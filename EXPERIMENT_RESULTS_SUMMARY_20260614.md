# LoCoRec 实验结果汇总（2026-06-14）

## 1. 使用说明

当前结果来自三个不同阶段，不能直接混为同一组最终实验：

1. **跨数据集旧协议结果**：Office、Beauty、Sports、Toys and Games，使用早期 LC-SoftSID/CRSID 固定融合实现，可用于说明方法演进和模块有效性。
2. **最终层次门控旧协议结果**：仅 Office，使用 Prior-guided Attention 和 Hierarchical Residual Gate。
3. **修复后协议结果**：仅 Office，修复了训练负采样、完整历史屏蔽、Transformer padding mask 和邻居 tie-break。该结果是当前 `LoCoRec` 代码的有效主结果。

因此，论文最终主表应等待 Beauty、Sports、Toys 在修复后协议下重跑。下表中的跨数据集旧结果不应与修复后的 Office 结果放在同一张最终主表中。

## 2. 当前最终结果（修复后协议）

| Dataset | Model | NDCG@5 | HR@5 | NDCG@10 | HR@10 | NDCG@20 | HR@20 | Valid NDCG@10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Office | LoCoRec | 0.055617 | 0.080122 | **0.067797** | **0.117839** | **0.082381** | **0.175739** | 0.083931 |

配置：`dim=128`、`max_len=50`、`delta=3`、`M=4`、`H=50`、Prior-guided Attention、Hierarchical Residual Gate、`seed=2026`。

来源：`runs/office/locorec_protocol_fixed_20260614/test_metrics.json`。

## 3. 跨数据集总体结果（旧协议）

| Dataset | Method | NDCG@10 | HR@10 | NDCG@20 | HR@20 |
|---|---|---:|---:|---:|---:|
| Office | SASRec | 0.061245 | 0.106014 | 0.075210 | 0.162080 |
| Office | Hard CRSID | 0.066010 | 0.115189 | 0.079431 | 0.168807 |
| Office | LC-SoftSID | **0.067432** | **0.118654** | **0.080910** | **0.172069** |
| Beauty | SASRec | 0.044830 | 0.077047 | 0.052492 | 0.107454 |
| Beauty | Hard CRSID | 0.051285 | **0.090283** | **0.060418** | **0.126593** |
| Beauty | LC-SoftSID | **0.051396** | 0.089478 | 0.060372 | 0.125117 |
| Sports | SASRec | 0.023956 | 0.042587 | 0.029039 | 0.062813 |
| Sports | Hard CRSID | 0.028490 | 0.051295 | 0.034350 | 0.074583 |
| Sports | LC-SoftSID | **0.028916** | **0.052784** | **0.034492** | **0.075060** |
| Toys and Games | SASRec | 0.062765 | 0.099485 | 0.070554 | 0.130458 |
| Toys and Games | Hard CRSID | 0.069450 | 0.117501 | 0.079050 | 0.155485 |
| Toys and Games | LC-SoftSID | **0.069823** | **0.118921** | **0.079333** | **0.156727** |

LC-SoftSID 相对 SASRec 的 NDCG@10 提升分别为：Office `10.10%`、Beauty `14.65%`、Sports `20.71%`、Toys and Games `11.24%`。

需要注意，Beauty 中 Soft SID 相比 Hard CRSID 仅在 NDCG@10 上小幅提升 `0.22%`，HR 指标略低；该数据集不能宣称 Soft SID 在所有指标上稳定优于 Hard CRSID。

来源：`runs/paper_results/main_comparison.csv`。

## 4. 跨数据集模块消融（旧固定融合版本）

### 4.1 NDCG@10

| Variant | Beauty | Sports | Toys and Games |
|---|---:|---:|---:|
| Full LC-Soft | **0.051396** | **0.028916** | **0.069823** |
| Hard CRSID | 0.051285 | 0.028490 | 0.069450 |
| w/o Shared Residual | 0.047690 | 0.028059 | 0.068468 |
| w/o Private Residual | 0.034128 | 0.017110 | 0.048678 |
| SASRec | 0.044830 | 0.023956 | 0.062765 |

### 4.2 结论

- 移除 **private residual** 在三个数据集上均造成最大下降，说明物品级辨识信息不可缺失。
- 移除 **shared residual** 均导致下降，说明 SID 共享参数确实提供跨物品迁移。
- Soft SID 相对 Hard CRSID 的提升较小但在三个数据集的 NDCG@10 上方向一致。
- 该组实验使用早期 `delta=2` 固定融合实现，不等同于当前最终 LoCoRec。

来源：

- `runs/paper_results/beauty/module_ablation.csv`
- `runs/paper_results/sports/module_ablation.csv`
- `runs/paper_results/toys_games/module_ablation.csv`

## 5. Office 核心消融（Prior-guided 固定分配版本）

| Variant | Valid NDCG@10 | NDCG@10 | HR@10 | NDCG@20 | HR@20 |
|---|---:|---:|---:|---:|---:|
| Full LC-SoftCRSID | **0.081967** | **0.067309** | 0.119062 | **0.081255** | 0.174516 |
| Hard SID (`M=1`) | 0.081115 | 0.066743 | 0.118858 | 0.081035 | **0.175535** |
| w/o Prior Bias | 0.081643 | 0.065448 | 0.111723 | 0.080267 | 0.170642 |
| w/o Shared Residual | 0.080242 | 0.066761 | **0.120897** | 0.079215 | 0.170438 |
| w/o Private Residual | 0.075191 | 0.063961 | 0.115189 | 0.078154 | 0.171865 |
| Earlier Learnable Allocation | 0.080506 | 0.063138 | 0.113354 | 0.077563 | 0.170438 |

该表支持：候选 Soft SID、局部先验、shared residual 和 private residual 均有作用。但该 Full 仍是固定分配版本，不是当前层次门控最终模型。

来源：`runs/office/lcsoft_core_ablation_20260613/summary.csv`。

## 6. 候选构造与注意力探索（Office）

| Experiment | Valid NDCG@10 | Test NDCG@10 | HR@10 | 结论 |
|---|---:|---:|---:|---|
| SID-overlap (`delta=3`) | 0.080506 | 0.063138 | 0.113354 | 优于文本 kNN |
| Text-kNN | 0.079065 | 0.061812 | 0.110296 | 未超过多槽 SID 邻域 |
| Prior-guided Attention | **0.081967** | **0.067309** | **0.119062** | 当前候选加权方案 |
| Candidate learned without prior | 0.081334 | 0.066530 | 0.118247 | 去除局部先验后下降 |

多槽 SID 邻域相对 text-kNN 的 NDCG@10 提升约 `2.15%`；Prior-guided 相对无先验候选学习提升约 `1.17%`。

## 7. 门控方案演进（Office，旧协议）

| Fusion | Valid NDCG@10 | NDCG@10 | HR@10 |
|---|---:|---:|---:|
| Flat three-way gate | 0.080488 | 0.062610 | 0.109888 |
| Regularized three-way gate | 0.081599 | 0.064668 | 0.111927 |
| Hierarchical residual gate (`scale=0.5`) | 0.081154 | 0.067162 | 0.113558 |
| Hierarchical residual gate (`scale=0.3`) | **0.081192** | **0.067530** | **0.114985** |

层次门控明显优于平坦三路竞争。`scale=0.3` 是进入当前 LoCoRec 的最终门控配置。

## 8. Office 冷启动与长尾结果（旧协议）

以下冷启动结果基于 Prior-guided 固定分配 Full 模型，而非修复后最终 LoCoRec。

### 8.1 LoCoRec 与 SASRec

| Frequency group | Count | LoCoRec NDCG@10 | SASRec NDCG@10 | Relative gain |
|---|---:|---:|---:|---:|
| Overall | 4905 | 0.067309 | 0.057078 | 17.92% |
| Cold `0-5` | 1392 | **0.029050** | 0.013488 | **115.38%** |
| `f=0` | 66 | **0.034271** | 0.000000 | -- |
| `f=1-2` | 460 | 0.009917 | 0.005034 | 97.00% |
| `f=3-5` | 866 | 0.038815 | 0.019007 | 104.21% |
| `f=6-10` | 741 | 0.034675 | 0.020151 | 72.08% |
| `f>10` | 2772 | 0.095245 | 0.088838 | 7.21% |

这组结果是当前最强的长尾证据：相对提升主要集中在低频物品，而头部物品仍保持正向收益。

### 8.2 冷启动模块作用

| Variant | Cold `0-5` NDCG@10 | 相对 Full 变化 |
|---|---:|---:|
| Full | **0.029050** | -- |
| Hard SID | 0.029004 | -0.16% |
| w/o Prior Bias | 0.028775 | -0.95% |
| w/o Shared Residual | 0.022418 | **-22.83%** |
| w/o Private Residual | 0.030238 | +4.09% |
| Earlier Learnable Allocation | 0.028282 | -2.64% |

解释：shared residual 对冷启动最关键；private residual 对整体和头部辨识重要，但在极低频区域可能引入未充分训练的私有参数。因此当前最终代码对 `f=0` 物品屏蔽 private residual，并使用层次门控控制其强度。

来源：`runs/office/cold_start_core_with_sasrec_20260614/cold_start_metrics.csv`。

## 9. 最终层次门控的冷启动证据（旧协议）

| Model | Overall NDCG@10 | Cold `0-5` NDCG@10 | Warm `>5` NDCG@10 |
|---|---:|---:|---:|
| Hierarchical gate (`scale=0.3`) | **0.067530** | **0.036257** | 0.079922 |
| Fixed allocation | 0.067309 | 0.029050 | **0.082469** |

层次门控将 Cold `0-5` NDCG@10 从 `0.029050` 提升到 `0.036257`，相对提升约 `24.81%`；同时 Warm 组下降约 `3.09%`。这符合将方法重点转向冷启动/弱监督物品的叙述，但应如实报告整体与头部之间的权衡。

来源：`runs/office/hierarchical_residual_gate_scale030_20260614/cold_start_comparison/cold_start_metrics.csv`。

## 10. 尚缺实验

1. 修复后协议下 Beauty、Sports、Toys and Games 的最终 LoCoRec 主结果。
2. 修复后协议下对应的 SASRec 公平基线。
3. 修复后协议下四个数据集的冷启动分桶结果。
4. 当前最终层次门控版本的核心消融，需要统一使用相同训练轮数和评估协议重跑。
5. 若论文声称多随机种子结果，需要补齐至少三个种子；现有主要结果多数为 `seed=2026`。

