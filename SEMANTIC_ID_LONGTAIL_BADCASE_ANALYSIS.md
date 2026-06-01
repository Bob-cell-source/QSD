# Semantic ID 长尾与 Badcase 分析

本文档记录对 Office 数据集 badcase 的重新分析。结论重点从“长尾 item”转向“Semantic ID 本身的长尾效应”：少数热门 semantic token / slot code 覆盖大量 item，并在错误推荐中形成 semantic hub。

## 1. Semantic ID 分布高度不均衡

Office 数据集共有 2420 个 item，每个 item 有 4 个 Semantic ID slot。

### Slot-level 热门 code

第 2 槽最明显：

```text
slot2 code 49: 591 items, 24.42%
slot2 code 43: 503 items, 20.79%
slot2 code 31: 420 items, 17.36%
```

也就是说，仅 slot2 的前三个 code 就覆盖了 62.57% 的 item。

第 4 槽也存在明显 hub：

```text
slot4 code 497: 476 items, 19.67%
slot4 code 276: 91 items, 3.76%
slot4 code 0:   63 items, 2.60%
```

全体 slot-token 维度上：

```text
top 5 slot-token share:  22.05%
top 10 slot-token share: 27.61%
top 20 slot-token share: 35.68%
top 50 slot-token share: 53.20%
```

这说明 Semantic ID 空间并不是均匀共享，而是少数 token 承担了大量 item 的语义连接。

## 2. Badcase 中热门 Semantic ID 更像 semantic hub

以 `runs/office/bad_cases_slot_overlap2_review.json` 为例：

```text
base_correct_comp_wrong:
  target avg_train = 39.37
  target avg_overlap_group_size = 27.86
  target avg_hot_slots = 2.27 / 4
  wrong top avg_hot_slots = 2.29 / 4

both_wrong:
  target avg_train = 17.65
  target avg_overlap_group_size = 29.07
  target avg_hot_slots = 2.36 / 4
  wrong top avg_hot_slots = 2.30 / 4
```

其中 `avg_hot_slots` 表示一个 item 的 4 个 Semantic ID slot 中，有多少 slot 落在该槽位 top 10% 高频 code 内。

这说明错误并不只发生在低频 item。即使 target 的训练频次不低，只要它处在高共享 semantic token 连接区域，也可能被热门语义主题带偏。

## 3. 错误 Top-K 被少数语义主题反复占据

在 `bad_cases_slot_overlap2_review.json` 中，错误 Top-K 里反复出现的 item 和主题包括：

```text
Quartet / dry-erase / board / marker
Wilson Jones / binder
Swingline / stapler
Mead / notebook / paper
Scotch / tape
Epson / printer
```

这些并不一定是全局最高频 item，而是通过热门 Semantic ID token 连接起来的高密度语义区域。

典型错误 Top-K 热门 slot-code：

```text
slot2 code 49
slot4 code 497
slot2 code 43
slot2 code 31
```

这些 code 在错误候选中反复出现，说明模型容易把“共享热门语义码”当成足够强的用户意图证据。

## 4. 典型现象

### 4.1 语义泛化变成 semantic shortcut

例如目标是 HP / Epson / Brother 这类墨盒、打印机、扫描仪时，历史中确实存在相邻办公设备证据。但热门 token 会进一步把候选拉向泛办公设备、打印机生态、甚至无关的办公耗材。

这不是简单的 item long-tail，而是 semantic token sharing 过宽导致的 shortcut。

### 4.2 强 exact evidence 有时被过度校准破坏

在 `bad_cases_binary_vs_learnable_overlap2.md` 中，Wilson Jones binder、Brother toner、HP ink 等样本有明确同系列或几乎相同 Semantic ID 证据。

Binary Evidence 能利用 exact evidence，而 learnable reliability 有时会把注意力转向更泛化的 binder/printer/office theme。

这说明热门 semantic token 不只是会制造假阳性，也会稀释真正可靠的强证据。

### 4.3 both_wrong 更像“item tail + semantic hub”叠加

`both_wrong` 的 target 平均训练频次更低：

```text
bad_cases_slot_overlap2_review both_wrong avg_train = 17.65
reliability_ablation qsd_vs_learnable both_wrong avg_train = 16.15
binary_vs_learnable both_wrong avg_train = 15.89
```

但它们的 semantic overlap 和 hot-slot 数量仍然较高。因此失败机制更准确地说是：

```text
low item evidence + high semantic hub exposure
```

而不是单纯“低频 item 学不好”。

## 5. 对方法设计的启发

后续方法不应再强调普通 item long-tail，而应强调：

```text
Semantic ID space has its own long-tail distribution.
Head semantic tokens dominate sharing and create semantic hubs.
Tail semantic evidence is easily drowned by high-frequency semantic codes.
```

更合适的方法方向是：

1. 不把所有 shared semantic token 当作同等证据。
2. 区分 head semantic token 与 tail semantic token 的作用。
3. 对热门 semantic token 做 residualization / de-bias，而不是简单降低权重。
4. 让协同信息修正 Semantic ID 的热门 token shortcut，而不是仅仅和 Semantic ID late fusion。

这比“长尾 item 推荐”更贴近当前 badcase 观察。
