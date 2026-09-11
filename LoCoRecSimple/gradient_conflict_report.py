"""Report the audit honestly, including occupancy-preserving null controls."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
import torch

from .gradient_conflict_diagnostic import write_json


def occupancy_null(seed_dir, rows, permutations=200):
    """Shuffle gradient identities within level/frequency strata, preserving groups.

    This checks finite-support/rank-tie effects and frequency composition. It is
    not a test of causal harm, and the exchangeability assumption is not proven.
    """
    output = seed_dir / 'occupancy_null.json'
    if output.exists():
        return json.loads(output.read_text())
    values = dict(np.load(seed_dir / 'gradients.npz'))
    settings = json.loads((seed_dir / 'manifest.json').read_text())
    counts = values['counts']
    eligible = np.where((counts >= settings['args']['min_events']).all(0))[0]
    index = np.full(counts.shape[1], -1, dtype=int)
    index[eligible] = np.arange(len(eligible))
    frequency = np.bincount([i for row in rows for i in row['items'][:-2]], minlength=counts.shape[1])
    bins = np.digitize(frequency[eligible], np.unique(np.quantile(frequency[eligible], [.25,.5,.75])))
    groups = [np.where(bins == b)[0] for b in np.unique(bins)]
    with (seed_dir / 'tokens.csv').open() as f:
        token_rows = list(csv.DictReader(f))
    result = {}
    for mode in ('full', 'positive'):
        pairs = np.load(seed_dir / f'pairs_{mode}.npz')['pairs']
        tokens, inverse = np.unique(pairs[:,1].astype(int), return_inverse=True)
        info = {int(r['token']): r for r in token_rows if r['mode'] == mode}
        occupancy = np.array([int(info[t]['occupancy']) for t in tokens])
        levels = np.array([int(info[t]['level']) for t in tokens])
        observed = np.array([float(info[t]['conflict_rate']) for t in tokens])
        pair_count = np.bincount(inverse, minlength=len(tokens))
        average = (values[mode] * counts[:,:,None,None]).sum(0) / np.maximum(counts.sum(0)[:,None,None], 1)
        blocks = []
        for level in np.unique(pairs[:,0]).astype(int):
            take = np.where(pairs[:,0] == level)[0]
            grad = average[eligible,level-1]
            unit = grad / np.maximum(np.linalg.norm(grad,axis=1,keepdims=True),1e-30)
            blocks.append((take, unit @ unit.T, index[pairs[take,2].astype(int)], index[pairs[take,3].astype(int)]))
        rng = np.random.default_rng(8181)
        null_rhos, null_level = [], {str(l): [] for l in np.unique(levels)}
        for _ in range(permutations):
            negative = np.empty(len(pairs), dtype=float)
            for take, cosine, left, right in blocks:
                permutation = np.arange(len(eligible))
                for group in groups:
                    permutation[group] = rng.permutation(group)
                negative[take] = cosine[permutation[left], permutation[right]] < 0
            rate = np.bincount(inverse, weights=negative, minlength=len(tokens)) / pair_count
            null_rhos.append(float(spearmanr(occupancy, rate).statistic))
            for level in np.unique(levels):
                keep = levels == level
                null_level[str(level)].append(float(spearmanr(occupancy[keep],rate[keep]).statistic))
        def describe(observed_rho, samples):
            finite = np.asarray(samples)[np.isfinite(samples)]
            if not len(finite):
                return {'observed_rho': observed_rho, 'valid_permutations': 0}
            return {'observed_rho': observed_rho, 'valid_permutations': len(finite),
                    'null_mean': float(finite.mean()), 'null_95_interval': np.quantile(finite,[.025,.975]).tolist(),
                    'fraction_null_at_least_observed': float((1+(finite >= observed_rho).sum())/(len(finite)+1))}
        result[mode] = {'pooled': describe(float(spearmanr(occupancy,observed).statistic),null_rhos),
                        'by_level': {str(l): describe(float(spearmanr(occupancy[levels==l],observed[levels==l]).statistic),null_level[str(l)]) for l in np.unique(levels)}}
    write_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', default='runs/office/gradient_conflict_diagnostic_20260910')
    parser.add_argument('--sensitivity-dir', default='runs/office/gradient_conflict_simple_sensitivity_20260910')
    args = parser.parse_args()
    root = Path(args.run_dir)
    summary = json.loads((root / 'summary.json').read_text())
    first_manifest = json.loads((root / f'seed{next(iter(summary))}' / 'manifest.json').read_text())
    rows = json.loads((Path(first_manifest['args']['dataset_dir']) / 'sequences.json').read_text())
    nulls = {}
    for seed in summary:
        nulls[seed] = occupancy_null(root / f'seed{seed}', rows)
    write_json(root / 'occupancy_null_summary.json', nulls)
    lines = ['# Hard SID 梯度冲突诊断', '',
             '完成当前 consolidated Hard/Soft 三种子配对诊断。此报告不把负夹角等同于有害迁移，也不将描述性相关性解释为因果作用。', '',
             '## 范围与定义', '',
             'Office 共2420个物品，913个在两组不同训练用户中各有至少4个目标样本；各组每物品最多16个样本。共184个token有至少4个合格物品，其中140个token有至少20个测试目标用于Soft增益分析。结果不覆盖所有稀疏物品。', '',
             '`L_i` 为下一物品目标是i的训练样本平均CE。full梯度包含历史、正候选与负候选路径；positive仅保留同一损失的正候选表示路径。使用训练好的、按验证集选择的Hard checkpoint，训练参数不更新。', '',
             '## 主要数值', '',
             '| Seed | full负夹角比例 | 两用户组均冲突 | positive负夹角比例 | 占用量–冲突ρ(full) | 冲突–Soft增益ρ(full) |',
             '|---|---:|---:|---:|---:|---:|']
    for seed,d in summary.items():
        f,p = d['modes']['full'], d['modes']['positive']
        lines.append(f"| {seed} | {f['within_token_pair_conflict_rate']:.2%} | {f['within_token_stable_pair_conflict_rate']:.2%} | {p['within_token_pair_conflict_rate']:.2%} | {f['occupancy_conflict_rho']:.3f} | {f['conflict_gain_rho']:.3f} |")
    lines += ['', '“两用户组均冲突”要求两组cosine均<−0.01；其他负夹角比例以cosine<0计。比例按token内物品对条目加权，同一物品对可能属于多层，不能作为独立样本做显著性检验。', '',
              '## 占用量关联的置换审计', '',
              '在每层内按训练频次四分位分层置换梯度身份，保持每个token的总占用、合格物品数、配对数及频次构成。每个seed/mode做200次。空分布用于检查有限组规模及秩并列效应，不是因果检验。', '',
              '| Seed | Mode | 实际ρ | 置换平均ρ | 置换95%区间 | 置换≥实际的比例（加一校正） |',
              '|---|---|---:|---:|---|---:|']
    for seed, dd in nulls.items():
        for mode, data in dd.items():
            x=data['pooled']
            lines.append(f"| {seed} | {mode} | {x['observed_rho']:.3f} | {x['null_mean']:.3f} | [{x['null_95_interval'][0]:.3f}, {x['null_95_interval'][1]:.3f}] | {x['fraction_null_at_least_observed']:.3f} |")
    lines += ['', '## 语义相近、行为支持不同的物品对', '',
              '语义相近取同token物品对的文本cosine最高四分位。行为有两个独立于SID构造的观测：训练前驱物品分布和共同训练用户。零重合只能表示未观察到共同支持，不能等同于真实偏好完全不同。', '',
              '| Seed | 行为定义 | 无重合冲突率(full) | 高重合冲突率(full) | 两组条目数 |',
              '|---|---|---:|---:|---|']
    for seed,d in summary.items():
        for name,b in d['modes']['full']['semantic_behavior'].items():
            lines.append(f"| {seed} | {name} | {b['zero_overlap_conflict_rate']:.2%} | {b['high_overlap_conflict_rate']:.2%} | {b['zero_overlap_pairs']}/{b['high_overlap_pairs']} |")
    lines += ['', '## Soft收益和Hard性能', '',
              'Soft收益采用相同模型结构、相同seed和协议的test_per_user配对差值；token分组依据目标物品的固定Hard SID。三个seed的全局冲突–收益ρ均接近零且为正，不支持预设的明显负相关。逐层方向混合，控制占用、频次、测试支持和梯度范数后也没有统一方向。详细系数保存在各seed的summary.json。', '',
              '高冲突token也没有在所有层/seed上对应更低的Hard NDCG，不能把冲突热点直接称为性能失败位置。token组在不同层间重叠；本轮不给出假定组独立的p值。', '',
              '## 图与原始数据', '']
    for seed in summary:
        lines.append(f'- Seed {seed}: [PNG](seed{seed}/diagnostic.png) / [PDF](seed{seed}/diagnostic.pdf) / [token表](seed{seed}/tokens.csv)。')
    sensitivity = Path(args.sensitivity_dir) / 'summary.json'
    if sensitivity.exists():
        lines += ['', '## 简单单共享表 Hard/Soft 的敏感性验证', '',
                  '另在无固定private系数、无公共bias的simple Hard/Soft上重复seed2026诊断。该对照独立报告，不能和主模型混为三种子结果。']
        for seed,d in json.loads(sensitivity.read_text()).items():
            for mode,m in d['modes'].items():
                lines.append(f"- {seed}/{mode}: 负夹角={m['within_token_pair_conflict_rate']:.2%}；稳定冲突={m['within_token_stable_pair_conflict_rate']:.2%}；占用–冲突ρ={m['occupancy_conflict_rho']:.3f}；冲突–Soft增益ρ={m['conflict_gain_rho']:.3f}。")
    lines += ['', '## 可以与不可以声称什么', '',
              '- 可以报告：有一部分可跨训练用户组复现的梯度不一致；应同时给出总体规模、路径敏感性和覆盖率。',
              '- 占用关联需要连同置换基线一起解释，不能仅凭正Spearman系数证明高占用引起冲突。',
              '- 序列前驱上下文提供部分行为证据，但共同用户口径不一致，不能泛化为所有协同行为定义。',
              '- 当前没有证据将梯度冲突直接连接到Soft共享失败或拆分收益。保留该空结果，不能为了叙事筛选token或seed。', '']
    (root / 'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(nulls), flush=True)


if __name__ == '__main__':
    main()
