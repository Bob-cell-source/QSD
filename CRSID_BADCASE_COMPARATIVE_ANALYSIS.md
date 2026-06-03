# CRSID / QSD / SASRec Badcase 对比分析

本文档分析当前 CRSID 相比之前 QSD 和 SASRec 的 badcase 变化，重点回答三个问题：

```text
1. CRSID 改进了哪些 QSD / SASRec 的错误？
2. CRSID 又在哪些样本上变差？
3. 这些现象说明当前方法解决了什么，还没有解决什么？
```

分析依据包括：

```text
runs/office/bad_cases_qsd_vs_crsid_overlap2.json
runs/office/bad_cases_qsd_vs_crsid_overlap2.md
runs/office/bad_cases_sasrec_vs_qsd_lowfreq_more.json
runs/office/bad_cases_sasrec_vs_qsd_highsharing_more.json
runs/office/bad_cases_sasrec_vs_qsd_overlap2_more.json
runs/office/crsid_module_ablation/summary.csv
```

并补充全量 test set 上 SASRec / QSD / CRSID 的 hit@10 组合统计。组合 key 顺序为：

```text
SASRec / QSD / CRSID
1 = target in Top-10
0 = target not in Top-10
```

## 1. 全量命中组合

| Key | 含义 | Count |
|---|---|---:|
| `000` | 三者都错 | 4021 |
| `001` | 只有 CRSID 对 | 144 |
| `010` | 只有 QSD 对 | 154 |
| `011` | QSD 和 CRSID 对，SASRec 错 | 74 |
| `100` | 只有 SASRec 对 | 101 |
| `101` | SASRec 和 CRSID 对，QSD 错 | 107 |
| `110` | SASRec 和 QSD 对，CRSID 错 | 55 |
| `111` | 三者都对 | 249 |

从数量上看：

```text
CRSID 命中样本数 = 001 + 011 + 101 + 111 = 568
QSD 命中样本数   = 010 + 011 + 110 + 111 = 532
SASRec 命中样本数 = 100 + 101 + 110 + 111 = 512
```

这和整体指标一致：

```text
CRSID NDCG@10 = 0.067048
QSD   NDCG@10 = 0.060348
SASRec NDCG@10 = 0.058792
```

CRSID 整体最好，但不是所有 QSD 或 SASRec 的正确样本都被保留。它主要新增了两类收益：

```text
001: CRSID 独自救回的样本，144 个。
101: SASRec 和 CRSID 对、QSD 错的样本，107 个。
```

第二类尤其重要：它说明 CRSID 不只是“比 QSD 语义更强”，而是能修复一部分 QSD 的语义漂移错误。

## 2. CRSID 相比 QSD 改好的 badcase

在 `bad_cases_qsd_vs_crsid_overlap2` 中，`comp_correct_base_wrong` 表示：

```text
QSD 错，CRSID 对
```

这些样本大致可以分成三类。

### 2.1 修复 QSD 的热门泛语义漂移

典型样本：

```text
target: HP 940XL C4907AN#140 Ink Cartridge in Retail Packaging-Cyan
sid = [1, 52, 140, 388]
train_count = 3
prefix_group_size = 1
overlap_group_size = 5
```

历史中有：

```text
HP 940XL Black    sid=[1,43,140,497]
HP 940XL Yellow   sid=[1,43,140,74]
HP 940XL Magenta  sid=[1,43,140,497]
```

这个 case 在早期分析里是 QSD 的典型失败：target 在 prefix 上是孤立点，QSD 的 semantic branch 没法通过 prefix 直接借力，反而被 Expo marker、Post-it labels、folder、tape 等热门办公主题吸走。

CRSID 的 top item：

```text
1. HP 940XL Cyan target
2. Lamy Bottle Ink
3. Noodlers Ink
4. Brother digital color printer
5. Expo marker
```

这里 CRSID 改好的原因不是它修复了 Semantic ID。target 仍然是 prefix_group_size=1，说明 SID 构建错配还在。真正改进来自表示方式变化：

```text
QSD:
  score = id_score + semantic_score
  semantic branch 会把用户历史中的热门语义主题直接加到候选分数上。

CRSID:
  不再额外加 semantic score。
  Semantic ID 只参与 item representation。
  低频 target 的 private residual 权重较小，shared semantic residual 仍可从共享 token 借力；
  同时 unified dot-product 避免 semantic branch 单独把热门办公簇推得过高。
```

因此，CRSID 修复的是 QSD 的“语义过度外推”：QSD 把历史中更强的办公主题推上去，CRSID 则更倾向于保留与 target 相关的墨水/打印耗材表示。

### 2.2 对同系列 / 同用途商品的局部语义更稳

典型样本：

```text
target: Lamy Refill Converter (Z24)
sid = [0, 43, 203, 29]
train_count = 6
prefix_group_size = 11
overlap_group_size = 29
```

历史中有：

```text
Noodlers Ink
Lamy Safari Converter Z24
Zebra / uni-ball pens
```

QSD top 中出现 Pilot G2、Swingline staples、Scotch tape 等偏泛化办公主题，target 未进 Top-10。CRSID top 中 target 排第 2，并且 top 列表包含：

```text
Pilot Iroshizuku bottled ink
Lamy Refill Converter
Lamy Refills Converter
Pilot fountain pen
Private Reserve ink
```

这说明 CRSID 对“钢笔/墨水/转换器”这个局部使用语义保持更稳定。原因是 shared semantic residual 和 private residual 被合并进 item representation 后，模型学习的是 item 与用户序列表征的统一相似度，而不是额外 semantic branch 对若干历史主题分别加分。

### 2.3 对高共享但有明确品牌/系列线索的样本更聚焦

典型样本：

```text
target: Epson WorkForce Pro WF-4640
sid = [61, 19, 232, 157]
train_count = 5
prefix_group_size = 10
overlap_group_size = 11
```

历史中有：

```text
Epson WorkForce WF-2540
Epson WorkForce Pro WF-4630
Canon Office printer
Epson ink cartridge
```

CRSID top：

```text
Epson WorkForce WF-3640
Epson WorkForce Pro WF-4640 target
Epson WorkForce WF-3620
Epson WorkForce WF-7620
Epson WorkForce WF-7610
```

QSD top 中混入 Avery label、Five Star cup、3M monitor arm、Fellowes laminator 等语义漂移候选。

这个 case 说明：当 target 有足够局部共享邻居时，CRSID 的 shared residual 能保留系列信息；同时 private residual 和序列 encoder 让候选不至于被泛办公 token 拉走。

## 3. CRSID 相比 QSD 变差的 badcase

在 `bad_cases_qsd_vs_crsid_overlap2` 中，`base_correct_comp_wrong` 表示：

```text
QSD 对，CRSID 错
```

这些样本也有比较清晰的类型。

### 3.1 QSD 的额外 semantic score 能直接召回，CRSID 反而不够强

典型样本：

```text
target: Pilot G2 Retractable Premium Gel Ink Roller Ball Pens
sid = [52, 4, 219, 468]
train_count = 49
prefix_group_size = 26
overlap_group_size = 29
```

QSD 能把 target 排进 Top-10，但 CRSID 错。历史里有 BIC pen、Epson cartridge、Epson printer、stapler、phone 等混合主题。

这个样本的问题是：target 本身并不低频，prefix_group_size 也较大。QSD 的 semantic score 可以强行增强“pen / writing instrument”语义区域；CRSID 则把语义变成 embedding basis 和 residual 后，语义召回强度被 unified representation 稀释了。当用户历史主题复杂时，CRSID 可能更偏向序列中其他更强的协同主题。

换句话说：

```text
QSD 的优点：语义分支召回力强，能硬拉高同语义簇 candidate。
CRSID 的代价：去掉额外 semantic score 后，某些需要强语义召回的样本会变弱。
```

### 3.2 对中高频、强 item-specific 的商品，CRSID 可能丢失精确记忆

典型样本：

```text
target: Quartet Chalkboard
train_count = 53
prefix_group_size = 6
overlap_group_size = 30
```

QSD 把 target 排第 7，CRSID 错。CRSID top 被 Swingline acrylic organizer、Quartet notepad、Wilson Jones sheet protector 等替代。

这个现象说明 CRSID 仍然有“表示平滑”的副作用。虽然 `alpha_i = freq/(freq+tau)` 对中高频 item 会提高 private residual 权重，但 item embedding 仍然由 semantic basis + residual 组成，不是纯 ID embedding。对于一些 QSD 能通过 ID/semantic score 偶然命中的样本，CRSID 的表示分解可能把它拉向更平滑的局部邻域。

### 3.3 shared residual 对热门 overlap token 仍有残余吸引

典型样本：

```text
target: Canon Photo Paper Plus Glossy II
train_count = 51
prefix_group_size = 3
overlap_group_size = 102
```

overlap_group_size 很大，说明 target 共享了高频 token。QSD 能命中，CRSID 错，但 CRSID top 仍然是 Canon ink / printer / photo paper 附近的候选。

这里不是完全语义漂移，而是细粒度 disambiguation 不够：CRSID 能留在 Canon / printer / ink 的局部区域，但没有把具体 target 排进 Top-10。原因是当前 CRSID 的 shared residual 是对多个 SID token 直接 mean pooling：

```text
r_shared = mean(E_shared(z1), E_shared(z2), E_shared(z3), E_shared(z4))
```

它没有区分哪个 token 是热门泛化 token，哪个 token 是真正区分 Canon photo paper 的细粒度 token。因此在 overlap 很大的样本上，CRSID 仍然可能被热门 token 的共享 residual 稀释。

## 4. CRSID 相比 SASRec 改好的地方

CRSID 相比 SASRec 的主要收益来自两个组合：

```text
001: 只有 CRSID 对，144 个
011: QSD 和 CRSID 对，SASRec 错，74 个
```

也就是 CRSID 至少救回了 218 个 SASRec 错误样本。

### 4.1 低频但有可共享语义邻居的 item

典型样本：

```text
target: Epson WorkForce Pro WF-4640
train_count = 5
prefix_group_size = 10
overlap_group_size = 11
```

SASRec top 中有若干 Epson printer，但 target 没进 Top-10；CRSID 把 target 排第 2。原因是低频 target 的 ID embedding 训练不足，但 Semantic ID shared residual 可以从 WorkForce 系列和 printer 相关 token 借力。

这正对应最初 QSD 成功案例的机制，但 CRSID 以更干净的方式实现：

```text
不是额外 semantic score 拉分；
而是 target item embedding 本身通过 shared residual 更接近同系列商品。
```

### 4.2 Semantic ID prefix 错配时，CRSID 有时能靠 overlap token 和 residual 救回

HP 940XL Cyan 是最关键例子：

```text
target prefix_group_size = 1
overlap_group_size = 5
train_count = 3
```

严格说它仍然是 SID mismatch：Cyan 没有和 Black / Yellow / Magenta 进入同一个 prefix group。但 CRSID 在全量统计中能命中，说明它不是只依赖 prefix，而是通过所有 slot 的 token embedding 和 residual 表示获得了少量可迁移信号。

这也是当前 CRSID 相比“prefix-only semantic branch”的优势：它的 `semantic_pool` 对所有 slot 做 pooling，不只看前两个 prefix。

## 5. CRSID 相比 SASRec 变差的地方

CRSID 也丢了一部分 SASRec 能命中的样本：

```text
100: 只有 SASRec 对，101 个
110: SASRec 和 QSD 对，CRSID 错，55 个
```

这类样本说明：有些情况下纯 ID 记忆或序列协同关系比 SID 表示更可靠。

### 5.1 强 item-specific 记忆被语义表示平滑

典型样本：

```text
target: Westcott Axis iPoint Evolution Electric Heavy Duty Pencil Sharpener
train_count = 38
prefix_group_size = 13
overlap_group_size = 17
```

SASRec 能命中，但 QSD 和 CRSID 都错。这说明 target 的正确推荐可能来自纯 ID 序列共现记忆，而不是 Semantic ID 共享。CRSID 虽然有 private residual，但仍然把 item 表示约束在 `semantic basis + residual` 结构下，相比纯 ID embedding 少了一些自由度。

### 5.2 用户历史有明确局部重复，但 SID 结构没有对齐

典型样本：

```text
target: Five Star Xpanz Zipper Pencil Pouch
train_count = 6
prefix_group_size = 3
overlap_group_size = 20
```

SASRec 和 QSD 对，CRSID 错。历史里有多个 Five Star / notebook / storage 类 item。CRSID top 里也出现 Five Star Magnetic Storage Pocket 等邻居，但 target 没进 Top-10。

这说明 CRSID 能进入正确大类，但细粒度 item 区分不足。对于 low-frequency 但用户历史有明确 item-level 关联的样本，纯 ID 序列模式有时比 SID 共享更精确。

## 6. 分组层面的变化

### 6.1 频次分布

命中组合的平均 train_count：

| Key | 含义 | Avg Train Count |
|---|---|---:|
| `001` | 只有 CRSID 对 | 25.26 |
| `010` | 只有 QSD 对 | 34.58 |
| `100` | 只有 SASRec 对 | 33.50 |
| `101` | SASRec 和 CRSID 对，QSD 错 | 30.05 |
| `110` | SASRec 和 QSD 对，CRSID 错 | 48.87 |
| `111` | 三者都对 | 36.52 |
| `000` | 三者都错 | 18.94 |

观察：

```text
只有 CRSID 对的样本平均频次低于只有 QSD 对和只有 SASRec 对。
SASRec/QSD 对但 CRSID 错的 110 样本平均频次最高。
```

这说明 CRSID 的新增收益更偏向中低频 item，而它损失的样本更多是中高频、需要精确 item-specific 记忆的样本。

这和方法机制一致：

```text
低频 item: alpha 小，更多使用 shared semantic residual。
高频 item: alpha 大，但仍然受 semantic basis 约束，可能不如纯 ID 完全自由。
```

### 6.2 Overlap sharing 分布

只有 CRSID 对的样本在各 overlap bucket 中都有分布：

```text
<=5:   6
6-10:  8
11-20: 53
21-50: 58
>50:  19
```

这说明 CRSID 不只对 high-sharing item 有效，也能救回一部分低 overlap 样本。但三者都错的样本仍然大量集中在各个 bucket，尤其是 `21-50` 和 `11-20`。

解释：

```text
CRSID 缓解了“有共享邻居但 ID 记不住”的问题；
但没有彻底解决“SID 共享结构本身错配”或“热门 token 过载”的问题。
```

## 7. 回到你最初的三个创新目标

### 7.1 目标 1：平衡语义信息和 ID 信息

当前 CRSID 已经比较好地解决了这个目标。

证据：

```text
完整 CRSID > SASRec > QSD 的部分 case 被修复。
完整 CRSID > QSD semantic-score baseline。
固定 alpha、只用 private、只用 shared 都不如完整 CRSID。
```

Badcase 层面也能看到：

```text
CRSID 修复了 HP 940XL、Lamy converter、Epson WorkForce 这类 QSD 语义漂移或低频召回不足样本。
```

结论：

```text
CRSID 确实比 QSD 更自然地平衡了语义泛化和 ID 记忆。
```

### 7.2 目标 2：解决 Semantic ID 里的长尾 / popularity bias

当前 CRSID 只解决了一部分。

已解决：

```text
item-level 长尾：
低频 item 通过 shared semantic residual 获得迁移；
HP 940XL、Epson WorkForce 这类低频 target 能被救回。
```

尚未解决：

```text
token-level 长尾 / hubness：
热门 SID token 仍然会稀释细粒度 item 表示；
overlap_group_size 很大的 Canon / Avery / Post-it 类样本仍有细粒度错排。
```

当前 `crsid_semhub` 的结果不理想，说明简单用 item-level semantic hubness alpha 不够。问题更可能需要 token-level 的表示修正或 tokenizer 层面的 popularity-aware assignment。

### 7.3 目标 3：Semantic ID 不匹配，补充用户侧信息

当前 CRSID 基本没有解决这个目标。

HP 940XL Cyan 虽然被 CRSID 救回，但它的 SID mismatch 仍然存在：

```text
Cyan:    [1,52,140,388], prefix_group_size=1
Black:   [1,43,140,497]
Yellow:  [1,43,140,74]
Magenta: [1,43,140,497]
```

CRSID 是在错误 SID 上尽量利用已有 token overlap，并没有修正 Semantic ID。Five Star / Locker / Magnetic Storage 这类样本也说明：用户历史里的 item-level 共现关系和 SID token 结构并不总是一致。

因此，如果论文要继续强化第三个创新点，不能只靠当前 CRSID。需要在 SID 构建阶段引入用户侧共现信息，或者做 behavior-consistent SID refinement。

## 8. 总结：CRSID 改进了什么，哪里还差

### 改进了什么

```text
1. 修复了 QSD 的一部分语义漂移。
   QSD 的 semantic score 容易把候选拉向热门办公主题；
   CRSID 把语义内化为 item representation，减少了额外语义分支的过度拉分。

2. 保留了低频 item 的语义共享收益。
   低频 target 的 private residual 权重较小，可以通过 SID shared residual 借力。

3. 比 SASRec 更能处理有语义邻居的长尾 item。
   只有 CRSID 对的样本平均 train_count 更低，说明它确实补足了纯 ID 记忆不足。

4. 比 QSD 更好地统一 ID 与 semantic。
   它不是 score-level fusion，而是 representation-level fusion。
```

### 还差在哪里

```text
1. 对需要强语义召回的样本，CRSID 有时不如 QSD。
   因为 QSD 的 semantic score 可以直接硬拉候选，而 CRSID 的语义强度被 embedding 表示吸收后更温和。

2. 对中高频、强 item-specific 的样本，CRSID 有时不如 SASRec。
   semantic basis 会带来表示平滑，可能损失纯 ID embedding 的精确记忆。

3. 对 SID mismatch 无法根治。
   HP 940XL、Five Star 等 case 的根因是 content SID 与用户侧商品系列结构不一致。

4. 对 token-level popularity bias 解决不彻底。
   当前 alpha 是 item-level 标量，不能区分同一个 item 内哪些 token 是热门 hub，哪些 token 是细粒度有效 token。
```

## 9. 后续改进方向

如果继续沿着 CRSID 改，建议不要再加 semantic branch 或 gate，而是改两个更根本的位置。

### 9.1 Token-level CRSID

当前 shared residual 是：

```text
r_shared = mean(E_shared(z1), E_shared(z2), E_shared(z3), E_shared(z4))
```

可以改成 token-level adaptive residual：

```text
r_shared = sum_l w_{i,l} E_shared(z_{i,l})
```

其中 `w_{i,l}` 不是用户 gate，而是 token 自身的可靠性：

```text
w_{i,l} = function(token frequency, token purity, token co-occurrence consistency)
```

这样能解决：

```text
热门 token 不应该和细粒度 token 等权；
overlap_group_size 很大的样本需要降低 hub token 的影响。
```

### 9.2 Behavior-consistent Semantic ID refinement

对 HP 940XL / Five Star 这类 case，根因是 SID 构建错配。需要在 SID 构建时加入用户侧共现：

```text
content similarity + behavior co-occurrence consistency
```

目标不是在推荐模型里加用户门控，而是让 SID 本身更符合商品系列和用户使用语义。

例如：

```text
HP 940XL Cyan / Black / Yellow / Magenta
应该共享稳定的 series token，而不是只共享少数零散 slot。
```

这会更直接对应你的第三个创新点。

## 10. 可写进论文的简短结论

```text
Badcase analysis shows that QSD benefits from Semantic ID sharing on low-frequency items,
but its auxiliary semantic branch often over-amplifies popular semantic regions and causes semantic drift.
CRSID alleviates this issue by internalizing Semantic IDs into item representation and adaptively balancing
private item residuals with shared Semantic-ID residuals. As a result, CRSID recovers many low-frequency or
series-related items that QSD or SASRec miss, such as HP ink cartridges, Lamy converters, and Epson WorkForce printers.
However, CRSID still fails when the Semantic ID structure itself is mismatched or when popular SID tokens dominate
fine-grained item distinctions. This suggests that future improvement should focus on token-level residual weighting
and behavior-consistent Semantic ID refinement rather than adding another scoring branch.
```
