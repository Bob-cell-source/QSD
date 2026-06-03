# CRSID 方法说明

本文档按照当前代码实现梳理 CRSID 方法。对应代码主要位于：

```text
qsdrec/model.py
  SASRecDynamicEncoder
  CRSIDRec

qsdrec/train.py
  build_semantic_table
  build_train_item_frequency
  build_semantic_hubness
  train

scripts/train_crsid.py
scripts/train_crsid_semhub.py
scripts/run_office_crsid_module_ablations.sh
```

## 1. 方法动机

原 QSDRec / EviQSD 的思路是在 SASRec 推荐分数之外继续增加 semantic branch、multi-interest query、evidence gate、hub penalty、contrastive semantic score 等模块。这类方法可以引入语义信息，但结构逐渐变复杂，并且语义信息更多体现在额外打分分支或后处理校准上。

CRSID 的目标是将 Semantic ID 直接纳入 item embedding 的构造过程，而不是把 Semantic ID 作为独立的额外打分分支。方法核心是将 item 表示分解为：

$$
e_i = b(z_i) + r_i^{\mathrm{coll}}
$$

其中 collaborative residual 再拆成：

$$
r_i^{\mathrm{coll}}
= \alpha_i r_i^{\mathrm{private}}
+ (1-\alpha_i) r_i^{\mathrm{shared}}
$$

直观上：

- `semantic basis` 提供由 item 语义簇决定的基础表示。
- `private item residual` 提供 item 自己的协同过滤残差，适合交互充分的头部 item。
- `shared semantic residual` 提供 Semantic ID token 级别共享的协同残差，适合交互稀疏的长尾 item。
- `alpha_i` 控制 private 与 shared residual 的比例，使不同 item 可以使用不同强度的协同残差来源。

最终 CRSID 仍然使用单一推荐分数：

$$
s(u,i) = h_u^\top e_i
$$

这里没有额外 semantic score 分支，也没有证据门控或 hub penalty。Semantic ID 的作用已经内化进 `e_i`。

## 2. 输入与数据划分

训练入口读取三个核心文件：

```text
{dataset_dir}/sequences.json
{dataset_dir}/stats.json
{semantic_ids}
```

`sequences.json` 中每个用户是一条按时间排序的 item 序列。`NextItemDataset` 使用 leave-one-out 风格划分：

```text
train:
  对每个用户序列 items，构造多条 next-item 样本：
  (items[:idx], items[idx]), idx = 1 ... len(items)-3

valid:
  输入 items[:-2]，预测 items[-2]

test:
  输入 items[:-1]，预测 items[-1]
```

训练样本会被左侧 padding 到 `max_len`。验证和测试阶段使用 full-ranking evaluation，并屏蔽用户历史中已经出现过的 item。

## 3. Semantic ID Table

CRSID 依赖已有 Semantic ID 文件。训练入口通过 `build_semantic_table` 读取：

```python
semantic_table, item_semantic_ids, num_semantic_tokens = build_semantic_table(...)
```

Semantic ID 文件中每个 item 对应多个槽位的离散 code：

$$
i \mapsto z_i = [z_{i,1}, z_{i,2}, \ldots, z_{i,L}]
$$

当前实验一般是 4 个槽位。代码会给不同槽位加 offset，使不同槽位的 code 不共享同一个 embedding id。例如第 1 槽和第 2 槽即使原始 code 都是 7，也会映射到不同 token id。这样可以避免不同层级/槽位的 code 在 embedding 表中发生语义混淆。

最终得到：

$$
\mathrm{semantic\_table}[i]
= [\mathrm{offset}(z_{i,1}), \mathrm{offset}(z_{i,2}), \ldots, \mathrm{offset}(z_{i,L})]
$$

padding item 为 0，对应 Semantic ID 也为 0。所有 embedding 都设置 `padding_idx=0`，并在初始化后将第 0 行置零。

## 4. CRSID Item 表示

CRSID 的核心函数是：

```python
CRSIDRec.item_representation(items)
```

给定 item id 张量，模型动态构造 item embedding，而不是直接查一个固定 `item_emb` 表。完整表达式为：

$$
e_i =
\mathrm{LayerNorm}
\left(
b(z_i)
+ \lambda
\left[
\alpha_i r_i^{\mathrm{private}}
+ (1-\alpha_i) r_i^{\mathrm{shared}}
\right]
\right)
$$

其中：

- \(b(z_i)\)：semantic basis。
- \(r_i^{\mathrm{private}}\)：private item residual。
- \(r_i^{\mathrm{shared}}\)：shared semantic residual。
- \(\lambda\)：residual scale，对应 `--cr-residual-scale`。
- \(\alpha_i\)：private/shared residual mixture coefficient。

最后模型会对 `e_i` 使用 dropout，并将 padding item 的表示置零。

### 4.1 Semantic Basis

semantic basis 使用一张 Semantic ID token embedding 表：

```python
self.semantic_basis_emb = nn.Embedding(num_semantic_tokens + 1, dim, padding_idx=0)
self.basis_proj = nn.Linear(dim, dim)
```

对 item 的所有 Semantic ID token 做 masked mean pooling：

$$
\mathrm{Pool}_b(z_i)
=
\frac{1}{L_i}
\sum_{\ell=1}^{L_i}
E_b(z_{i,\ell})
$$

再经过线性投影：

$$
b(z_i)
=
W_b \, \mathrm{Pool}_b(z_i)
$$

这一部分表达 item 的语义基础位置。共享同类 Semantic ID token 的 item 会在 basis 中自然接近，从而缓解纯 ID embedding 对长尾 item 学习不足的问题。

### 4.2 Shared Semantic Residual

shared semantic residual 使用另一张独立 embedding 表：

```python
self.semantic_residual_emb = nn.Embedding(num_semantic_tokens + 1, dim, padding_idx=0)
```

它同样对 Semantic ID token 做 masked mean pooling：

$$
r_i^{\mathrm{shared}}
=
\frac{1}{L_i}
\sum_{\ell=1}^{L_i}
E_s(z_{i,\ell})
$$

注意它和 semantic basis 不共享参数。二者职责不同：

- semantic basis 表达 item 的语义基础表示；
- shared semantic residual 表达 Semantic ID token 上共享的协同过滤残差信息。

shared residual 的意义是：如果一个长尾 item 自身交互少，模型仍可通过它的 Semantic ID token 从同语义簇的其他 item 中获得协同信号。

### 4.3 Private Item Residual

private item residual 使用 item 级 embedding 表：

```python
self.item_residual_emb = nn.Embedding(num_items + 1, dim, padding_idx=0)
```

即：

$$
r_i^{\mathrm{private}} = E_p(i)
$$

它保留每个 item 独有的协同过滤偏移，适合交互充分、具有稳定用户行为模式的 item。对于头部 item，过度依赖共享语义残差可能会损失 item 特有偏好，因此需要 private residual。

### 4.4 Adaptive Residual Mixture

private 与 shared residual 的混合由 `alpha_i` 控制：

$$
r_i =
\alpha_i r_i^{\mathrm{private}}
+ (1-\alpha_i) r_i^{\mathrm{shared}}
$$

当 `alpha_i` 接近 1 时，模型主要使用 item 私有残差；当 `alpha_i` 接近 0 时，模型主要使用 Semantic ID token 共享残差。

当前代码支持三种 `alpha_i` 来源：

```text
1. item-frequency alpha
2. semantic-hubness alpha
3. fixed alpha override，用于消融
```

## 5. Item-Frequency Alpha

`crsid` 版本使用 item 训练频次计算 alpha。

训练入口先统计 item 在训练历史中的出现次数：

```python
item_frequency = build_train_item_frequency(sequences, num_items)
```

统计范围为：

```python
for item in row["items"][:-2]:
    freq[item] += 1
```

即只使用每个用户 valid/test target 之前的历史，不使用验证集 target 和测试集 target。

alpha 公式为：

$$
\alpha_i
=
\frac{f_i}{f_i+\tau}
$$

其中 `tau` 对应：

```text
--cr-tail-tau
```

性质：

$$
\begin{aligned}
f_i = 0 &\Rightarrow \alpha_i = 0, \\
f_i \ll \tau &\Rightarrow \alpha_i \approx 0, \\
f_i \gg \tau &\Rightarrow \alpha_i \approx 1.
\end{aligned}
$$

其中 \(\alpha_i \approx 0\) 时偏向 shared semantic residual，\(\alpha_i \approx 1\) 时偏向 private item residual。

例如 `tau=20`：

$$
\begin{aligned}
f_i=1   &\Rightarrow \alpha_i=0.0476, \\
f_i=10  &\Rightarrow \alpha_i=0.3333, \\
f_i=20  &\Rightarrow \alpha_i=0.5000, \\
f_i=100 &\Rightarrow \alpha_i=0.8333.
\end{aligned}
$$

该版本的解释重点是 item 长尾自适应：头部 item 使用更多私有协同信息，长尾 item 使用更多语义共享协同信息。

## 6. Semantic-Hubness Alpha

`crsid_semhub` 版本不使用 item 训练频次，而是使用 Semantic ID token hubness 计算 alpha。

训练入口通过 `build_semantic_hubness` 统计每个 Semantic ID token 覆盖了多少 item：

```python
semantic_token_hubness, semantic_item_hubness = build_semantic_hubness(...)
```

token hubness 的计算为：

$$
c(z)
=
\sum_i \sum_{\ell=1}^{L_i}
\mathbf{1}[z_{i,\ell}=z]
$$

$$
H(z)
=
\frac{\log(1+c(z))}
{\max_{z'} \log(1+c(z'))}
$$

因此 `H(z)` 被归一化到 `[0, 1]`。在 `CRSIDRec.residual_alpha` 中，item 的 hubness 是其 Semantic ID token hubness 的 masked mean：

$$
\mathrm{hub}(i)
=
\frac{1}{L_i}
\sum_{\ell=1}^{L_i}
H(z_{i,\ell})
$$

然后计算：

$$
\alpha_i
=
\alpha_{\min}
+ (1-\alpha_{\min})
\cdot \mathrm{hub}(i)^\gamma
$$

对应参数：

```text
--cr-hub-alpha-floor
--cr-hub-alpha-gamma
```

解释：

- 如果 item 的 Semantic ID token 很热门，说明这些 token 覆盖大量 item，语义共享信号可能比较泛化甚至噪声更大，因此提高 `alpha_i`，更多依赖 private residual。
- 如果 item 的 Semantic ID token 较尾部，说明这些 token 更细粒度、更有辨识度，因此降低 `alpha_i`，更多保留 shared semantic residual。

该版本的解释重点是 Semantic ID token 长尾/头部 hub 自适应，而不是普通 item 频次长尾。

## 7. Fixed Alpha Override

当前代码新增了：

```text
--cr-alpha-override
```

如果该参数不为空，`CRSIDRec.residual_alpha` 会直接返回固定值：

$$
\alpha_i = \alpha_{\mathrm{override}}
$$

它会绕过 item-frequency alpha 和 semantic-hubness alpha。该开关主要用于模块消融，不建议作为主方法。

典型消融：

```text
--cr-alpha-override 0.5
  固定 private/shared 各占一半，用于证明 adaptive alpha 的必要性。

--cr-alpha-override 1.0
  只使用 private item residual。

--cr-alpha-override 0.0
  只使用 shared semantic residual。
```

## 8. Dynamic SASRec Encoder

普通 SASRec 使用内部 item embedding：

```python
x = self.item_emb(seq)
```

CRSID 的 item embedding 是动态构造的，因此新增：

```python
SASRecDynamicEncoder
```

其 forward 接收两个输入：

```python
encoder(seq, item_repr)
```

具体流程：

```text
1. 根据 seq 的 padding mask 构造 position id。
2. 使用传入的 item_repr 作为序列输入。
3. item_repr 乘以 sqrt(dim)，与 position embedding 相加。
4. 经过 dropout。
5. 使用 causal attention mask，保证当前位置不能看到未来 item。
6. 依次经过 MultiheadAttention 和 PointWiseFeedForward。
7. 每层之后用 seq.ne(0) 将 padding 位置置零。
8. 最后 LayerNorm。
9. 因为序列左 padding，最后一个位置就是最新非 padding item，返回 h[:, -1] 作为用户表示。
```

因此用户表示为：

$$
h_u =
\mathrm{SASRecDynamicEncoder}
\left(
e_{i_1}, e_{i_2}, \ldots, e_{i_t}
\right)
$$

除 item embedding 来源不同外，CRSID 的序列建模主体与 SASRec 保持一致。

## 9. Scoring

CRSID 对候选 item 也动态构造 embedding：

```python
cand_repr = item_representation(candidates)
```

然后使用点积打分：

```python
score = torch.einsum("bd,bcd->bc", h_id, cand_repr)
```

数学形式：

$$
s(u,i) = h_u^\top e_i
$$

`forward` 返回的字典中还包含若干兼容 QSDRec 训练框架的字段：

$$
\begin{aligned}
\mathrm{id\_score} &= s(u,i), \\
\mathrm{sem\_score} &= 0, \\
\mathrm{amateur\_sem\_score} &= 0, \\
\mathrm{hub\_loss} &= 0, \\
\mathrm{residual\_l2} &= \mathbb{E}\left[\left\|r_i^{\mathrm{private}}\right\|_2^2\right].
\end{aligned}
$$

其中 `sem_score`、`amateur_sem_score` 和 `hub_loss` 对 CRSID 没有实际作用，只是为了复用训练/评估代码接口。

## 10. Training Objective

默认训练目标为 sampled softmax。每个训练样本的候选集合格式为：

```text
[positive target, negative_1, negative_2, ...]
```

训练 loss 使用第 0 个候选作为正样本：

```python
labels = torch.zeros(scores.size(0), dtype=torch.long)
loss = cross_entropy(scores, labels)
```

对 CRSID：

$$
\mathcal{L}
=
\mathcal{L}_{\mathrm{sampled\_CE}}
+ \beta_{\mathrm{cr}}
\cdot \mathrm{residual\_l2}
$$

其中：

```text
--cr-residual-reg 默认 0.0
```

当前主实验中一般不启用 residual L2 正则。如果设置为正值，则只正则化候选 item 的 `item_residual_emb`，不正则化 semantic basis 或 shared residual。

训练框架也支持 `--train-objective full_softmax`，此时会对所有 item 构造分数并做 full softmax loss。虽然函数类型标注写的是 `QSDRec`，但 CRSID 的 `forward` 接口兼容，因此也可以运行。不过当前 CRSID probe 和消融默认使用 sampled objective。

## 11. Negative Sampling

训练时通过 `CandidateSampler` 生成负样本。候选集合总长度为：

$$
|\mathcal{C}|
=
1 + N_{\mathrm{hard}} + N_{\mathrm{random}}
$$

默认 CRSID probe 和模块消融使用：

```text
--num-hard-neg 0
--num-random-neg 100
```

即只使用随机负样本。代码也支持基于 Semantic ID 的 hard negative：

```text
--hard-neg-mode prefix
--hard-neg-mode overlap
```

但 CRSID 当前主设置关闭 hard negative，是为了保持方法结构简洁，并避免 hard negative 策略成为额外变量。

## 12. Evaluation

验证和测试都使用 full-ranking evaluation：

```python
evaluate_full_ranking(...)
```

评估流程：

```text
1. 对每个 batch 的用户序列得到用户表示。
2. 分块枚举所有 item，计算 full-ranking score。
3. 屏蔽用户历史中已经出现的 item。
4. 取 Top-K。
5. 计算 HR@5/10/20、Recall@5/10/20、NDCG@5/10/20。
```

验证阶段使用 `NDCG@10` 做 early stopping：

```text
best_valid_NDCG@10
```

训练结束后加载验证集最优 checkpoint，在 test set 上报告最终指标。结果保存到：

```text
{output_dir}/history.json
{output_dir}/test_metrics.json
{output_dir}/best.pt
```

## 13. 当前 CRSID 不包含的模块

CRSID 是独立于 QSDRec / EviQSD 的轻量方法。当前 `model_variant in {"crsid", "crsid_semhub"}` 时，不使用：

```text
semantic score branch
multi-interest query
semantic fusion weight
evidence gate
history overlap gate
reliability gate
hub reliability gate
strength / idf evidence gate
prior lift / mini lift
hub penalty
contrastive semantic score
diversity loss
disambiguation loss
```

虽然训练参数中仍然能看到 `--sem-weight`、`--dis-weight`、`--div-weight` 等 QSDRec 参数，但 CRSID 的 `forward` 只使用统一点积分数；这些 QSDRec 语义分支参数对 CRSID 主路径没有实际作用。

## 14. 命令行版本

### 14.1 Item-Frequency CRSID

可以使用独立入口：

```bash
python scripts/train_crsid.py \
  --dataset-dir runs/office \
  --semantic-ids runs/office/semantic_ids_rq.json \
  --output-dir runs/office/crsid_tau20_probe \
  --device cuda \
  --epochs 30 \
  --early-stop-patience 5 \
  --batch-size 256 \
  --max-len 50 \
  --dim 128 \
  --num-hard-neg 0 \
  --num-random-neg 100 \
  --cr-tail-tau 20 \
  --cr-residual-scale 1.0
```

等价于：

```bash
python scripts/train_qsdrec.py --model-variant crsid ...
```

### 14.2 Semantic-Hubness CRSID

可以使用独立入口：

```bash
python scripts/train_crsid_semhub.py \
  --dataset-dir runs/office \
  --semantic-ids runs/office/semantic_ids_rq.json \
  --output-dir runs/office/crsid_semhub_f005_g10_s10 \
  --device cuda \
  --epochs 30 \
  --early-stop-patience 5 \
  --batch-size 256 \
  --max-len 50 \
  --dim 128 \
  --num-hard-neg 0 \
  --num-random-neg 100 \
  --cr-hub-alpha-floor 0.05 \
  --cr-hub-alpha-gamma 1.0 \
  --cr-residual-scale 1.0
```

等价于：

```bash
python scripts/train_qsdrec.py --model-variant crsid_semhub ...
```

## 15. 模块级消融设计

模块消融脚本为：

```text
scripts/run_office_crsid_module_ablations.sh
```

运行：

```bash
bash scripts/run_office_crsid_module_ablations.sh
```

默认配置：

```text
PYTHON_BIN=/voice/bin/python
DATASET_DIR=runs/office
SEMANTIC_IDS=runs/office/semantic_ids_rq.json
OUT_ROOT=runs/office/crsid_module_ablation
DEVICE=cuda
EPOCHS=30
EARLY_STOP_PATIENCE=5
BATCH_SIZE=256
DIM=128
NUM_RANDOM_NEG=100
SEED=2026
```

可以通过环境变量覆盖：

```bash
EPOCHS=100 BATCH_SIZE=512 OUT_ROOT=runs/office/crsid_module_ablation_full \
bash scripts/run_office_crsid_module_ablations.sh
```

脚本会跳过已有 `test_metrics.json` 的实验。如果需要强制重跑：

```bash
FORCE=1 bash scripts/run_office_crsid_module_ablations.sh
```

### 15.1 实验组

```text
00_sasrec_id_only
  纯 ID SASRec baseline。
  作用：证明 CRSID 相比纯协同序列建模的收益。

01_qsdrec_semantic_score
  原 QSDRec semantic score baseline。
  作用：证明 CRSID 不是简单增加一个 semantic score branch。

10_crsid_full_tau20_s10
  完整 CRSID。
  组成：semantic basis + adaptive private/shared residual。

11_crsid_basis_only_no_residual
  设置 --cr-residual-scale 0。
  只保留 semantic basis。
  作用：证明 collaborative residual 是否必要。

12_crsid_no_semantic_basis
  设置 --cr-disable-semantic-basis。
  只保留 residual path。
  作用：证明 semantic basis 是否必要。

13_crsid_no_shared_residual
  设置 --cr-disable-shared-residual。
  去掉 Semantic ID token 共享协同残差。
  作用：证明 shared semantic residual 是否有效。

14_crsid_no_private_residual
  设置 --cr-disable-private-residual。
  去掉 item 私有残差。
  作用：证明 private item residual 是否有效。

15_crsid_fixed_alpha_050
  设置 --cr-alpha-override 0.5。
  固定 private/shared 各 50%。
  作用：证明 adaptive alpha 是否优于固定混合。

16_crsid_private_only_alpha_100
  设置 --cr-alpha-override 1.0。
  只用 private item residual。
  作用：对照纯 item residual 是否足够。

17_crsid_shared_only_alpha_000
  设置 --cr-alpha-override 0.0。
  只用 shared semantic residual。
  作用：对照纯 semantic shared residual 是否足够。

20_crsid_semhub_full_f005_g10_s10
  使用 crsid_semhub。
  作用：对照 item-frequency alpha 与 semantic-hubness alpha。
```

### 15.2 模块有效性的判读

如果完整 CRSID 优于 `11_crsid_basis_only_no_residual`，说明仅靠 Semantic ID basis 不足，协同残差有效。

如果完整 CRSID 优于 `12_crsid_no_semantic_basis`，说明 semantic basis 不只是辅助项，而是 item 表示的有效基础。

如果完整 CRSID 优于 `13_crsid_no_shared_residual`，说明 Semantic ID token 级共享协同残差有效，尤其可以支持长尾 item。

如果完整 CRSID 优于 `14_crsid_no_private_residual`，说明 item 私有残差有效，模型需要保留头部 item 的个性化协同信息。

如果完整 CRSID 优于 `15_crsid_fixed_alpha_050`，说明按 item 频次或 Semantic ID hubness 自适应调节 private/shared residual 比例是有效的。

如果完整 CRSID 同时优于 `16_crsid_private_only_alpha_100` 和 `17_crsid_shared_only_alpha_000`，说明 private residual 与 shared residual 的组合优于单一路径。

如果 `10_crsid_full_tau20_s10` 优于 `20_crsid_semhub_full_f005_g10_s10`，说明当前数据集上 item-frequency tail adaptation 更稳。如果 `20` 更好，则说明 Semantic ID token hubness 是更关键的自适应变量。

## 16. 论文写法建议

方法部分可以将 CRSID 概括为：

```text
We propose CRSID, a collaborative-residual Semantic ID representation method.
Instead of using Semantic IDs as an auxiliary scoring branch, CRSID constructs
each item embedding as a semantic basis plus an adaptive collaborative residual.
The residual is decomposed into a private item residual and a shared Semantic-ID
residual, with an item-specific coefficient controlling the mixture.
```

中文表述可以写为：

```text
本文提出 CRSID，将 Semantic ID 从额外打分分支转化为 item 表示构造的一部分。
具体地，CRSID 将 item embedding 分解为语义基底和协同残差，其中协同残差由
item 私有残差与 Semantic ID token 共享残差组成，并通过自适应系数控制二者比例。
该设计使头部 item 能保留 item-specific 协同信息，同时使长尾 item 能借助语义共享残差
获得更稳定的协同表示。
```

实验部分可以围绕三条主张组织：

```text
1. Compared with SASRec and QSDRec semantic-score baseline, CRSID improves
   recommendation by internalizing Semantic ID into item representation.

2. Removing semantic basis, shared residual, private residual, or adaptive alpha
   hurts performance, showing that each component contributes to the final method.

3. Comparing item-frequency alpha and semantic-hubness alpha reveals whether the
   main bottleneck is item long-tail sparsity or Semantic ID token hubness.
```

## 17. 当前方法边界

CRSID 当前仍然是 item-level residual mixture。即使 `crsid_semhub` 使用 Semantic ID token hubness 计算 alpha，最终 alpha 仍然是 item 级标量：

$$
\alpha_i \in \mathbb{R}
$$

它还没有做到每个 Semantic ID token 单独控制 residual，例如：

$$
e_i
=
b(z_i)
+ r_i^{\mathrm{head\_token}}
+ r_i^{\mathrm{tail\_token}}
$$

因此，如果后续 bad case 继续显示错误主要来自某些头部 Semantic ID token 的过度共享，可以进一步扩展为 token-level residual gating。但当前 CRSID 的优势是结构简洁、训练稳定、解释清楚，适合作为主方法候选。
