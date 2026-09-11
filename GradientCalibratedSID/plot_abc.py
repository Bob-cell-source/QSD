"""Create standalone A/B/C figures and verify partition geometry after diagnosis."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='runs/office/gcss_diagnostic_abc_20260910');args=p.parse_args()
    root=Path(args.root)
    a=json.loads((root/'A_pair_conflict.json').read_text())['1']['levels']
    b=json.loads((root/'B_summary.json').read_text())
    c=json.loads((root/'C_comparisons.json').read_text())
    fig,axes=plt.subplots(1,3,figsize=(16,4.5))
    x=np.arange(4);width=.25
    for n,(key,label) in enumerate([('same_token','Same SID token'),('semantic_neighbor_different_token','Semantic neighbor / different token'),('random_pair','Random pair')]):
        axes[0].bar(x+(n-1)*width,[100*r[key]['negative_fraction'] for r in a],width,label=label)
    axes[0].set(xticks=x,xticklabels=['L1','L2','L3','L4'],ylabel='Negative dot products (%)',title='A: Conflict exists, but compare controls')
    axes[0].legend(fontsize=7)
    for l,r in b.items():
        values=[q['hard_ndcg'] for q in r['quartiles_low_to_high']]
        axes[1].plot(range(1,len(values)+1),values,marker='o',label='L'+l)
    axes[1].set(xlabel='Compatibility quartile (low to high)',ylabel='Hard SID NDCG@10',xticks=[1,2,3,4],title='B: Does lower compatibility mean difficulty?')
    axes[1].legend()
    labels=[];values=[];lo=[];hi=[]
    for name in ('high_conflict','high_alignment'):
        for subset in ('all','affected'):
            r=c[name+'_gradient_minus_'+name+'_random'][subset]
            labels.append(name.replace('_',' ')+' / '+subset)
            values.append(r['delta']);lo.append(max(0,r['delta']-r['bootstrap95'][0]));hi.append(max(0,r['bootstrap95'][1]-r['delta']))
    axes[2].errorbar(values,np.arange(4),xerr=[lo,hi],fmt='o',capsize=3)
    axes[2].axvline(0,color='gray',linestyle='--');axes[2].set(yticks=np.arange(4),yticklabels=labels,xlabel='NDCG@10: guided minus random',title='C: Paired user bootstrap 95% interval')
    axes[2].tick_params(axis='y',labelsize=8)
    fig.tight_layout();fig.savefig(root/'ABC_diagnostics.png',dpi=180);fig.savefig(root/'ABC_diagnostics.pdf');plt.close(fig)
    protocol=json.loads((root/'protocol.json').read_text())
    seed_root=Path(protocol['source_checkpoint']).parent.parent
    g=torch.load(seed_root/'grad_signature.pt',weights_only=True)['grad_signature'].numpy()
    cp=torch.load(protocol['source_checkpoint'],weights_only=True,map_location='cpu')
    hard=cp['model']['item_encoder.hard_sid_table'][1:].numpy()
    geometry={}
    for path in sorted(root.glob('training/*/split.json')):
        details=json.loads(path.read_text())
        if not details:continue
        parent_values=[];within_values=[]
        for r in details:
            members=np.where(hard[:,r['level']-1]==r['token'])[0]
            moved=np.isin(members+1,r['moved_item_ids'])
            for mask,dest in [(np.ones(len(members),bool),parent_values),(moved,within_values),(~moved,within_values)]:
                v=g[members[mask]];dots=v@v.T
                dest.extend(dots[np.triu_indices(len(v),1)].tolist())
        summary=lambda v:{'pairs':len(v),'mean_dot':float(np.mean(v)),'negative_fraction':float((np.asarray(v)<0).mean())}
        geometry[path.parent.name]={'parent':summary(parent_values),'within_children':summary(within_values)}
    (root/'C_partition_geometry.json').write_text(json.dumps(geometry,indent=2))

if __name__=='__main__':main()
