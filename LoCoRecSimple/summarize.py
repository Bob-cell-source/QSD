"""Summarize only completed, matched seeds. Never select runs using test scores."""
import argparse
from pathlib import Path
import torch
from LoCoRec.locorec.io import read_json, write_json


def summarize(root, reference_roots=()):
    root=Path(root)
    rows=[]
    seen=set()
    for source in [root,*map(Path,reference_roots)]:
        for path in sorted(source.glob('*/seed*/result.json')):
            row=read_json(path)
            key=(row['variant'],row['seed'])
            if key in seen:raise ValueError(f'Duplicate model/seed: {key}')
            seen.add(key)
            rows.append((path.parent,row))
    if not rows:
        raise ValueError('No completed runs')
    for _,row in rows:
        for key in ('dataset_dir','semantic_ids','dim','max_len','negatives','lr','weight_decay','dropout','batch_size','epochs','patience','gate_warmup'):
            if row['args'][key]!=rows[0][1]['args'][key]:
                raise ValueError(f'Unmatched protocol field: {key}')
    variants=sorted({r['variant'] for _,r in rows})
    summary={}
    for variant in variants:
        group=[r for _,r in rows if r['variant']==variant]
        metrics={}
        for key in group[0]['test']:
            values=torch.tensor([r['test'][key] for r in group],dtype=torch.double)
            metrics[key]={'mean':float(values.mean()),'std':float(values.std()) if len(values)>1 else None}
        summary[variant]={'seeds':[r['seed'] for r in group],'metrics':metrics,
                          'parameters':group[0]['parameters'],'best_epochs':[r['best_epoch'] for r in group],
                          'elapsed_seconds':sum(r['elapsed_seconds'] for r in group)}
    comparisons={}
    for treatment,baseline in [('hard','id'),('soft','hard'),('full','soft'),('full','hard'),
                               ('full_fixed','full'),('full_global','full_fixed'),('full_equal','full_fixed'),
                               ('hard','hard_shuffled'),('hard_shuffled','id'),('id_transfer','id'),
                               ('id_transfer','id_transfer_shuffled'),
                               ('hard_scaled','id_scaled'),('soft_scaled','hard_scaled'),('full_scaled','soft_scaled'),
                               ('consolidated_scaled','full_scaled'),
                               ('consolidated_scaled','consolidated_hard_scaled'),
                               ('consolidated_scaled','consolidated_no_bias_scaled'),
                               ('consolidated_scaled','consolidated_uniform_scaled')]:
        t={r['seed']:(p,r) for p,r in rows if r['variant']==treatment}
        b={r['seed']:(p,r) for p,r in rows if r['variant']==baseline}
        seeds=sorted(set(t)&set(b))
        if not seeds:continue
        differences=[]
        for seed in seeds:
            a=torch.load(t[seed][0]/'test_per_user.pt',weights_only=True)
            c=torch.load(b[seed][0]/'test_per_user.pt',weights_only=True)
            if a.shape!=c.shape:raise ValueError('Unmatched evaluation users')
            differences.append(a-c)
        matrix=torch.stack(differences).double()
        means=matrix.mean(-1)
        comparisons[f'{treatment}_minus_{baseline}']={
            'seeds':seeds,'NDCG@10_per_seed_delta':means.tolist(),
            'mean_delta':float(means.mean()),'seed_delta_std':float(means.std()) if len(seeds)>1 else None,
            'positive_seed_count':int(means.gt(0).sum())}
    result={'models':summary,'paired_seed_comparisons':comparisons,
            'note':'Matched seeds only; std is sample std across training seeds. No test-based seed selection.'}
    write_json(root/'analysis.json',result)
    lines=['# 单共享表 LoCoRec 统一对照结果','',
           '所有版本从头训练，使用相同数据、encoder、逐 epoch 采样种子与全库评估。Full 保留原辅助损失，但统一关闭 item-side dropout。consolidated 从随机 Full 参数等价转换后开始独立训练，无训练后 checkpoint 输入；其消融保留其余初始张量。带 scaled 后缀的训练 logits 除以 sqrt(dim)。',
           '', '| Model | Seeds | NDCG@10 mean ± std | HR@10 mean ± std | Parameters | Best epochs |',
           '|---|---|---:|---:|---:|---|']
    order=['id','hard_shuffled','hard','soft','full','full_fixed','full_global','full_equal','id_transfer','id_transfer_shuffled']
    for v in order+[v for v in summary if v not in order]:
        if v not in summary:continue
        r=summary[v]
        def metric(key):
            x=r['metrics'][key]
            return f"{x['mean']:.5f}" + (f" ± {x['std']:.5f}" if x['std'] is not None else ' (single seed)')
        lines.append(f"| {v} | {r['seeds']} | {metric('NDCG@10')} | {metric('HR@10')} | {r['parameters']:,} | {r['best_epochs']} |")
    lines+=['','## 配对 seed 差值','']
    for name,r in comparisons.items():
        lines.append(f"- {name}: seeds={r['seeds']}, delta={r['NDCG@10_per_seed_delta']}, mean={r['mean_delta']:.6f}, positive seeds={r['positive_seed_count']}/{len(r['seeds'])}.")
    lines+=['','注意：该实验没有将模型参数量强行匹配，参数差异本身是简化目标的一部分。三 seed 仍不构成跨数据集有效性的证明；不能与旧论文或不同协议结果直接计算增益。','']
    (root/'REPORT.md').write_text('\n'.join(lines))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir',required=True)
    parser.add_argument('--reference-dir',action='append',default=[])
    args=parser.parse_args()
    print(summarize(args.run_dir,args.reference_dir))
