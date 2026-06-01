# CR-SID 实验版本说明

CR-SID 是一个独立于现有 QSDRec / EviQSD 的轻量方法版本，目标是避免继续堆叠 semantic branch、gate、hub penalty、contrastive 等模块。

当前保留两个互不覆盖的版本：

```text
crsid:
  item-frequency alpha，初始版本。

crsid_semhub:
  semantic-token hubness alpha，针对 Semantic ID token 长尾/头部 hub 的版本。
```

## 1. 方法核心

CR-SID 将 item 表示分解为：

```text
item representation = semantic basis + collaborative residual
```

其中 collaborative residual 又由两部分组成：

```text
tail-adaptive residual = alpha_i * private_item_residual
                       + (1 - alpha_i) * shared_semantic_residual
```

初始版本中，`alpha_i` 由训练历史中的 item 频次决定：

```text
alpha_i = freq_i / (freq_i + tau)
```

直观解释：

- 头部 item 交互多，更依赖自己的 private collaborative residual。
- 长尾 item 交互少，更依赖 Semantic ID token 共享的 collaborative residual。
- Semantic ID 不再作为额外打分分支，而是直接参与 item embedding 构造。

最终仍然使用统一打分：

```text
score(u, i) = h_u^T e_i
```

## 2. 当前代码实现

当前实现位置：

```text
qsdrec/model.py
  SASRecDynamicEncoder
  CRSIDRec

qsdrec/train.py
  --model-variant qsdrec|crsid
  build_train_item_frequency

scripts/train_crsid.py
  CR-SID 独立训练入口

scripts/train_crsid_semhub.py
  CR-SID Semantic-hub 独立训练入口
```

### 2.1 Semantic ID Table

训练入口首先读取已有 Semantic ID：

```python
semantic_table, item_semantic_ids, num_semantic_tokens = build_semantic_table(...)
```

`semantic_table[item]` 对应 item 的多槽位 Semantic ID：

```text
[z_1, z_2, z_3, z_4]
```

代码中会对不同 slot 加 offset，使不同槽位的 code 不共用同一个 embedding id。

### 2.2 Item Frequency Alpha

当前版本用训练历史中的 item 频次控制 private residual 与 shared residual 的比例：

```python
item_frequency = build_train_item_frequency(sequences, num_items)
```

该统计只使用每个用户序列的训练部分：

```python
for item in row["items"][:-2]:
    freq[item] += 1
```

不使用 valid/test target。

CRSIDRec 中的自适应系数为：

```text
alpha_i = freq_i / (freq_i + tau)
```

其中 `tau` 对应命令行参数：

```text
--cr-tail-tau
```

含义：

```text
freq_i 越高，alpha_i 越接近 1，更依赖 item-private residual。
freq_i 越低，alpha_i 越接近 0，更依赖 Semantic ID token shared residual。
```

例如 `tau=20` 时：

```text
freq=1   -> alpha=0.0476
freq=10  -> alpha=0.3333
freq=100 -> alpha=0.8333
```

### 2.2b Semantic Hubness Alpha

`crsid_semhub` 不再使用 item frequency 计算 alpha，而是使用 item 的 Semantic ID token hubness：

```text
hub(z_i) = mean(H(z_i1), H(z_i2), H(z_i3), H(z_i4))
```

其中 `H(z)` 来自全局 item 集上的 Semantic ID token 频率：

```text
H(z) = normalized log(1 + token_count)
```

代码中复用 `build_semantic_hubness` 生成的 `semantic_token_hubness`。

Semantic-hub alpha 为：

```text
alpha_i = hub_alpha_floor + (1 - hub_alpha_floor) * hub(z_i)^gamma
```

对应参数：

```text
--cr-hub-alpha-floor
--cr-hub-alpha-gamma
```

解释：

```text
Semantic ID token 越热门，alpha_i 越大，模型越依赖 private item residual，
避免热门 semantic token 的 shared residual 把泛办公主题带进来。

Semantic ID token 越偏尾部，alpha_i 越小，模型越保留 shared semantic residual，
让更有区分度的尾部 semantic token 提供共享协同信息。
```

### 2.3 Item Representation

核心函数是：

```python
CRSIDRec.item_representation(items)
```

它由三部分构成。

第一部分是 semantic basis：

```python
basis = basis_proj(pool(E_basis(z_i1), ..., E_basis(z_i4)))
```

第二部分是 shared semantic residual：

```python
shared_residual = pool(E_shared_residual(z_i1), ..., E_shared_residual(z_i4))
```

它表示 Semantic ID token 上共享的协同残差信息。

第三部分是 private item residual：

```python
private_residual = E_item_residual(i)
```

最终 item embedding 为：

```text
e_i = Norm(
        b(z_i)
        + residual_scale *
          [alpha_i * r_i_private + (1 - alpha_i) * r_i_shared]
      )
```

其中：

```text
b(z_i): semantic basis
r_i_private: item-level collaborative residual
r_i_shared: Semantic-ID-token shared collaborative residual
residual_scale: --cr-residual-scale
```

### 2.4 Dynamic SASRec Encoder

原 `SASRecEncoder` 内部直接查 `item_emb(seq)`。CR-SID 需要动态构造 item embedding，因此新增：

```python
SASRecDynamicEncoder
```

它接收已经构造好的 `item_repr`：

```python
seq_repr = item_representation(seq)
h_u, _ = encoder(seq, seq_repr)
```

除 item embedding 来源不同外，position embedding、causal mask、Transformer block 与原 SASRec 逻辑保持一致。

### 2.5 Scoring and Loss

候选 item 表示同样由 `item_representation(candidates)` 构造：

```python
cand_repr = item_representation(candidates)
score = einsum("bd,bcd->bc", h_u, cand_repr)
```

对应公式：

```text
score(u, i) = h_u^T e_i
```

CR-SID 当前没有：

```text
semantic branch
multi-interest query
evidence gate
hub penalty
contrastive semantic score
semantic fusion weight
```

训练损失在 `model_variant == "crsid"` 或 `model_variant == "crsid_semhub"` 时为：

```text
loss = sampled_cross_entropy + cr_residual_reg * residual_l2
```

默认：

```text
--cr-residual-reg 0
```

验证和测试仍然使用原来的 full-ranking evaluation。

## 3. 版本管理

现有 QSDRec 默认入口不变：

```bash
python scripts/train_qsdrec.py ...
```

CR-SID 使用独立入口：

```bash
python scripts/train_crsid.py ...
```

CR-SID Semantic-hub 使用独立入口：

```bash
python scripts/train_crsid_semhub.py ...
```

或者显式指定：

```bash
python scripts/train_qsdrec.py --model-variant crsid ...
python scripts/train_qsdrec.py --model-variant crsid_semhub ...
```

默认 `--model-variant qsdrec`，因此旧实验配置不会被改变。

## 4. Office Smoke / Probe

Item-frequency 版本 probe：

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

Semantic-hub 版本 probe：

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

## 5. 最小消融

第一轮只建议改 `tau` 和 residual scale：

```text
crsid_tau5_s10
crsid_tau20_s10
crsid_tau50_s10
crsid_tau20_s05
crsid_tau20_s20
```

不要同时打开 QSDRec 的 semantic weight、evidence gate、hub penalty 等机制，否则会失去 CR-SID 作为精简主方法的意义。

Semantic-hub 版本第一轮只建议改 `hub_alpha_floor`、`hub_alpha_gamma` 和 residual scale：

```text
crsid_semhub_f005_g10_s10
crsid_semhub_f010_g10_s10
crsid_semhub_f005_g05_s10
crsid_semhub_f005_g20_s10
crsid_semhub_f005_g10_s05
crsid_semhub_f005_g10_s20
```

## 6. 当前版本的局限

`crsid` 的 `alpha_i` 基于 item frequency，因此它更接近“item long-tail adaptive residual”。

但 badcase 重新分析显示，当前项目更关键的问题是：

```text
Semantic ID token 本身存在长尾和头部 hub。
少数热门 semantic token 覆盖大量 item，并在错误 Top-K 中反复出现。
```

`crsid_semhub` 已经将自适应变量从 item frequency 转向 semantic-token frequency / semantic hubness：

```text
alpha_i = alpha(z_i)
```

但它仍然是 item-level alpha。后续还可以进一步做 token-level residual：

```text
e_i = semantic basis
    + head-token debiased residual
    + tail-token collaborative residual
```

这样方法会更贴近 Semantic ID 长尾，而不是普通 item 长尾。
