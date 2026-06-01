# CR-SID 实验版本说明

CR-SID 是一个独立于现有 QSDRec / EviQSD 的轻量方法版本，目标是避免继续堆叠 semantic branch、gate、hub penalty、contrastive 等模块。

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

`alpha_i` 由训练历史中的 item 频次决定：

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

## 2. 版本管理

现有 QSDRec 默认入口不变：

```bash
python scripts/train_qsdrec.py ...
```

CR-SID 使用独立入口：

```bash
python scripts/train_crsid.py ...
```

或者显式指定：

```bash
python scripts/train_qsdrec.py --model-variant crsid ...
```

默认 `--model-variant qsdrec`，因此旧实验配置不会被改变。

## 3. Office Smoke / Probe

建议先跑一个小 probe，输出到新目录，避免覆盖旧版本：

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

## 4. 最小消融

第一轮只建议改 `tau` 和 residual scale：

```text
crsid_tau5_s10
crsid_tau20_s10
crsid_tau50_s10
crsid_tau20_s05
crsid_tau20_s20
```

不要同时打开 QSDRec 的 semantic weight、evidence gate、hub penalty 等机制，否则会失去 CR-SID 作为精简主方法的意义。
