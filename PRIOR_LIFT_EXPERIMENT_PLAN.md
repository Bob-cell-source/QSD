# Prior-Lift Evidence Experiment Plan

本文档记录对 R-EviQSD 的精简版实验：Popularity-Prior Corrected Semantic Evidence。

## 1. Motivation

R-EviQSD 的 learnable reliability 有效，但方法略显复杂。Prior-Lift 版本将 token reliability 收缩为一个显式统计量：

\[
Lift_u(z)=\log P(z|H_u)-\tau\log P(z)
\]

其中 \(P(z|H_u)\) 表示用户历史中对 token \(z\) 的个性化支持，\(P(z)\) 表示全局 SID token 流行先验。

如果一个 token 只是全局热门，即使它出现在用户历史里，lift 也不会过高；如果一个长尾 token 在用户历史中反复出现，它会获得更高 personalized evidence lift。

## 2. Implemented Variants

| Variant | 参数 | 说明 |
|---|---|---|
| `prior_lift` | `--evidence-gate prior_lift` | 对原始 SID token 计算 lift，并加到 semantic attention logits |
| `mini_lift` | `--evidence-gate mini_lift --mini-clusters ...` | 对高频 SID token 内部 mini-cluster 计算 lift |

Attention logits:

\[
\ell_{u,i,k,f}
=
q_{u,k}^{\top}t_{i,f}/\sqrt{d}
+\eta Lift_u(z_{i,f})
\]

Mini-cluster 版本：

\[
\ell_{u,i,k,f}
=
q_{u,k}^{\top}t_{i,f}/\sqrt{d}
+\eta Lift_u(c_{i,f})
\]

其中 \(c_{i,f}\) 是 token \(z_{i,f}\) 内部的 mini evidence cluster。

## 3. Why This Is Cleaner

相比 learnable reliability：

```text
same-slot evidence
cross-slot evidence
token specificity
latest support
MLP estimator
```

Prior-Lift 将它们收缩为：

```text
personalized evidence over global popularity prior
```

这更容易解释，也更贴近论文主命题：Semantic ID 的问题不是共享不足，而是共享证据被 popularity prior 污染。

## 4. Run Command

Office probe:

```bash
chmod +x scripts/run_office_prior_lift_probe.sh

PYTHON_BIN=python \
DATASET_DIR=runs/office \
SEMANTIC_IDS=runs/office/semantic_ids_rq.json \
EMBEDDINGS=runs/office/item_text_embeddings.npy \
EMBED_ITEM_IDS=runs/office/embedding_item_ids.json \
OUT_ROOT=runs/office/prior_lift_probe \
bash scripts/run_office_prior_lift_probe.sh
```

Beauty server probe:

```bash
PYTHON_BIN=python \
DATASET_DIR=runs/beauty \
SEMANTIC_IDS=runs/beauty/semantic_ids_rq.json \
EMBEDDINGS=runs/beauty/item_text_embeddings.npy \
EMBED_ITEM_IDS=runs/beauty/embedding_item_ids.json \
MINI_CLUSTERS=runs/beauty/mini_evidence_clusters.json \
OUT_ROOT=runs/beauty/prior_lift_probe \
BATCH_SIZE=512 \
EVAL_BATCH_SIZE=2048 \
EPOCHS=100 \
PATIENCE=10 \
bash scripts/run_office_prior_lift_probe.sh
```

## 5. Decision Rule

如果 `prior_lift` 或 `mini_lift` 能达到或超过 learnable reliability，则主方法建议改为：

```text
PL-EviQSD / MEL-SID
```

如果 prior-lift 明显低于 learnable reliability，但高于 QSD-base，则作为一个简洁可解释消融，证明 popularity prior correction 是有效组成部分。

如果 prior-lift 不如 Binary Evidence，则不进入主方法，只保留为 negative ablation。

