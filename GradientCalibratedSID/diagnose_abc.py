"""A/B/C diagnosis on the NEW pure Hard SID baseline, with matched interventions."""
import argparse
import copy
import csv
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

from LoCoRec.locorec.data import NextItemDataset, collate_eval
from GradientCalibratedSID.model import GCSS, load_model
from GradientCalibratedSID.trainer import fit


def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False))


def corr(x,y):
    if len(x)<3 or np.std(x)==0 or np.std(y)==0:
        return None
    return float(spearmanr(x,y).statistic)


def pair_stats(g, pairs):
    if not len(pairs):
        return {'pairs':0,'negative_fraction':None,'mean_dot':None}
    values = np.einsum('ij,ij->i',g[pairs[:,0]],g[pairs[:,1]])
    return {'pairs':len(pairs),'negative_fraction':float((values<0).mean()),
            'mean_dot':float(values.mean()),'negative_below_minus_001':float((values<-.01).mean())}


def experiment_a(hard,g,counts,neighbors):
    reports = {}
    for minimum in (1,5):
        eligible = counts>=minimum
        rows=[]
        rng=np.random.default_rng(7026+minimum)
        for level in range(hard.shape[1]):
            same=[]
            for token in np.unique(hard[:,level]):
                items=np.where((hard[:,level]==token)&eligible)[0]
                a,b=np.triu_indices(len(items),1)
                if len(a):same.append(np.column_stack([items[a],items[b]]))
            same=np.concatenate(same) if same else np.empty((0,2),int)
            semantic=set()
            for i, ns in enumerate(neighbors):
                if not eligible[i]:continue
                for jid in ns[ns>0]:
                    j=int(jid)-1
                    if eligible[j] and hard[i,level]!=hard[j,level]:semantic.add(tuple(sorted((i,j))))
            sem=np.array(sorted(semantic),dtype=int).reshape(-1,2)
            pool=np.where(eligible)[0]
            # Uniform ordered random pairs excluding self; sampling with replacement.
            a=rng.integers(len(pool),size=len(same)); b=rng.integers(len(pool)-1,size=len(same));b+=b>=a
            rand=np.column_stack([pool[a],pool[b]])
            rows.append({'level':level+1,'same_token':pair_stats(g,same),
                         'semantic_neighbor_different_token':pair_stats(g,sem),'random_pair':pair_stats(g,rand)})
        reports[str(minimum)]={'min_profile_count':minimum,'eligible_items':int(eligible.sum()),'levels':rows}
    return reports


@torch.no_grad()
def observations(model,loader,device,n):
    model.eval()
    vectors=torch.cat([model.encode_items(torch.arange(a,min(n+1,a+256),device=device)) for a in range(1,n+1,256)])
    result={k:[] for k in ('target','recall','ndcg','loss')}
    for history,target,known in loader:
        h,_=model.encode_sequence(history.to(device))
        scores=h@vectors.T
        for row,(seen,t) in enumerate(zip(known,target.tolist())):
            excluded=list(set(seen)-{t,0})
            scores[row,torch.tensor(excluded,dtype=torch.long,device=device)-1]=-torch.inf
        labels=target.to(device)-1
        loss=torch.nn.functional.cross_entropy(scores/math.sqrt(model.config['dim']),labels,reduction='none')
        # Match the training evaluator's top-20 then slice-10 exactly: SID
        # collisions yield exact ties, whose ordering can depend on topk k.
        top=scores.topk(min(20,n),-1).indices[:,:10]
        match=top.eq(labels[:,None]); hit=match.any(-1).float()
        ndcg=hit/torch.log2(match.float().argmax(-1).float()+2)
        for key,value in [('target',target),('recall',hit),('ndcg',ndcg),('loss',loss)]:result[key].append(value.cpu().numpy())
    return {k:np.concatenate(v) for k,v in result.items()}


def token_rows(hard,g,counts,frequency,obs):
    rows=[]
    for l in range(hard.shape[1]):
        for token in np.unique(hard[:,l]):
            members=np.where(hard[:,l]==token)[0]
            active=members[counts[members]>0]
            if len(active)<2:continue
            v=g[members].astype(np.float64)
            mean=float((np.square(v.sum(0)).sum()-np.square(v).sum())/(len(v)*(len(v)-1)))
            dot=v@v.T; pairs=dot[np.triu_indices(len(v),1)]
            sel=np.isin(obs['target']-1,members)
            row={'level':l+1,'token':int(token),'occupancy':len(members),'profiled_items':len(active),
                 'well_profiled_items':int((counts[members]>=3).sum()),'mean_compatibility':mean,
                 'conflict_rate':float((pairs<0).mean()),'mean_train_frequency':float(frequency[members].mean()),
                 'test_examples':int(sel.sum())}
            for k in ('recall','ndcg','loss'):row['hard_'+k]=float(obs[k][sel].mean()) if sel.any() else None
            rows.append(row)
    return rows


def cluster(items,g):
    """Two-means on unchanged, unnormalized signatures; no test outcomes used."""
    x=g[items].astype(np.float64)
    dist=np.square(x[:,None]-x[None,:]).sum(-1)
    a,b=np.unravel_index(dist.argmax(),dist.shape)
    centers=x[[a,b]].copy()
    labels=np.zeros(len(x),int)
    for _ in range(40):
        new=np.square(x[:,None]-centers[None]).sum(-1).argmin(-1)
        if len(np.unique(new))<2:
            # Degenerate signatures: deterministic balanced fallback, flagged later.
            new=np.arange(len(x))>=len(x)//2
        updated=np.stack([x[new==k].mean(0) for k in range(2)])
        if np.array_equal(labels,new):break
        labels=new.astype(int); centers=updated
    return labels.astype(int)


def choose_groups(rows,hard,g,budget=4):
    selected={'high_conflict':[],'high_alignment':[]}
    for l in range(hard.shape[1]):
        eligible=[r for r in rows if r['level']==l+1 and r['well_profiled_items']>=8 and r['occupancy']>=8]
        eligible.sort(key=lambda r:r['mean_compatibility'])
        half=len(eligible)//2
        low=eligible[:min(budget,half)]
        upper=eligible[half:]
        for r in low:
            selected['high_conflict'].append(r)
            match=min(upper,key=lambda s:abs(math.log(s['occupancy']/r['occupancy'])))
            selected['high_alignment'].append(match);upper.remove(match)
    return selected


def split_model(base,groups,g,randomized=False,seed=9026):
    model=copy.deepcopy(base)
    encoder=model.item_encoder
    old=encoder.sid_embedding.weight.detach()
    table=encoder.hard_sid_table.clone()
    additions=[]; details=[]
    rng=np.random.default_rng(seed)
    for group in groups:
        level=group['level']-1; token=group['token']
        items=torch.where(table[1:,level].eq(token))[0].cpu().numpy()
        labels=cluster(items,g)
        if randomized:labels=rng.permutation(labels) # Exactly same child sizes.
        new_id=len(old)+len(additions)
        additions.append(old[token].clone())
        table[torch.as_tensor(items[labels==1]+1,device=table.device),level]=new_id
        details.append({**group,'child_sizes':[int((labels==0).sum()),int((labels==1).sum())],
                        'new_token':new_id,'moved_item_ids':(items[labels==1]+1).tolist()})
    if additions:
        weights=torch.cat([old,torch.stack(additions)])
        encoder.sid_embedding=torch.nn.Embedding.from_pretrained(weights,freeze=False,padding_idx=0)
    encoder.hard_sid_table=table
    encoder.set_lookup(table.unsqueeze(-1),table.ne(0).float().unsqueeze(-1))
    with torch.no_grad():
        ids=torch.arange(len(table),device=table.device)
        torch.testing.assert_close(model.encode_items(ids),base.encode_items(ids),atol=0,rtol=0)
    return model,details


def paired_delta(new,base,mask):
    values=new['ndcg'][mask]-base['ndcg'][mask]
    if not len(values):return {'n':0,'delta':None,'bootstrap95':None}
    rng=np.random.default_rng(8026)
    means=[float(rng.choice(values,len(values),replace=True).mean()) for _ in range(1000)]
    return {'n':len(values),'delta':float(values.mean()),'bootstrap95':np.quantile(means,[.025,.975]).tolist()}


def report(root,rows,a,interventions,atomic=None):
    lines=['# Pure Hard SID: A/B/C diagnostics','',
           'Office, seed 2026. Positive-target directional means from frozen pure Hard SID. All observations below are diagnostic, not evidence of causal benefit unless intervention controls improve.','',
           '## A: Pair conflict','', '| Minimum observations | Level | Same token | Semantic neighbors / different token | Random |',
           '|---|---|---:|---:|---:|']
    for setting in a.values():
        for r in setting['levels']:
            fmt=lambda k:f"{100*r[k]['negative_fraction']:.3f}% (n={r[k]['pairs']})" if r[k]['pairs'] else 'NA'
            lines.append(f"| {setting['min_profile_count']} | {r['level']} | {fmt('same_token')} | {fmt('semantic_neighbor_different_token')} | {fmt('random_pair')} |")
    lines+=['','## B: Compatibility and difficulty','',
            'Groups require >=10 test examples. Quartiles are assigned within each SID level; each level is analyzed separately because an item belongs to one group at every level. CE uses the filtered full catalog, not 100 sampled negatives.','',
            '| Level | Groups | Spearman C vs NDCG | C vs Recall | C vs loss | C vs occupancy | C vs train frequency |',
            '|---|---:|---:|---:|---:|---:|---:|']
    b={}
    for level in sorted({r['level'] for r in rows}):
        group=[r for r in rows if r['level']==level and r['test_examples']>=10]
        x=[r['mean_compatibility'] for r in group]
        stats={k:corr(x,[r[k] for r in group]) for k in ('hard_ndcg','hard_recall','hard_loss','occupancy','mean_train_frequency')}
        if atomic is not None:
            stats['hard_minus_atomic_ndcg']=corr(x,[r['hard_minus_atomic_ndcg'] for r in group])
        lines.append('| '+str(level)+' | '+str(len(group))+' | '+' | '.join('NA' if v is None else f'{v:.4f}' for v in list(stats.values())[:5])+' |')
        bins=[]
        ordered=sorted(group,key=lambda r:r['mean_compatibility'])
        for part in np.array_split(np.arange(len(ordered)),4):
            rs=[ordered[i] for i in part]; total=sum(r['test_examples'] for r in rs)
            if not total:continue
            bins.append({'groups':len(rs),'test_examples':total,'mean_compatibility':float(np.mean([r['mean_compatibility'] for r in rs])),
                         **{k:sum(r[k]*r['test_examples'] for r in rs)/total for k in (('hard_ndcg','hard_recall','hard_loss','atomic_ndcg','hard_minus_atomic_ndcg') if atomic is not None else ('hard_ndcg','hard_recall','hard_loss'))}})
        b[str(level)]={'correlations':stats,'quartiles_low_to_high':bins}
    write(root/'B_summary.json',b)
    if atomic is not None:
        lines+=['','Atomic-ID is newly trained with the same backbone, negatives, validation selection and maximum epochs. Its parameter count differs from SID. Per-token Hard minus Atomic results are in token_groups.csv.']
    lines+=['','## C: Matched token splitting','',
            'Within each level select up to 4 lowest-compatibility groups with >=8 items having >=3 profile observations. High-alignment controls are selected from the upper half, matched on occupancy. "High conflict" means relative low compatibility, not necessarily majority negative pairs. Selection uses training signatures only. Random split preserves exact child sizes. Duplicated token rows preserve all initial item representations exactly. All continuation arms start from the same converged Hard checkpoint and reset AdamW; epoch 0 is eligible for validation selection.','',
            '| Arm | Test NDCG@10 | Best epoch | Parameters |','|---|---:|---:|---:|']
    for p in sorted(root.glob('training/*/result.json')):
        r=json.loads(p.read_text());lines.append(f"| {p.parent.name} | {r['test']['NDCG@10']:.6f} | {r['best_epoch']} | {r['parameters']} |")
    lines+=['','Paired user bootstrap intervals and affected/unaffected target subsets: C_comparisons.json. These intervals do not measure seed-to-seed training uncertainty. Single-seed and checkpoint-local interventions cannot establish a general causal conclusion.','']
    (root/'REPORT.md').write_text('\n'.join(lines))
    if rows:
        with (root/'token_groups.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',default='runs/office/gcss_20260910')
    parser.add_argument('--output',default='runs/office/gcss_diagnostic_abc_20260910')
    parser.add_argument('--seed',type=int,default=2026)
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args();torch.set_num_threads(2)
    source=Path(args.source);root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    cp=source/f'seed{args.seed}/stage1/hard_sid_checkpoint.pt'
    base,payload=load_model(cp,args.device);base.eval()
    config=payload['training_config'].copy();config['device']=args.device
    data=json.loads((Path(config['dataset_dir'])/'sequences.json').read_text())
    train=NextItemDataset(data,config['max_len'],'train')
    valid=DataLoader(NextItemDataset(data,config['max_len'],'valid'),batch_size=config['batch_size'],collate_fn=collate_eval)
    test=DataLoader(NextItemDataset(data,config['max_len'],'test'),batch_size=config['batch_size'],collate_fn=collate_eval)
    artifact=torch.load(source/'semantic_candidates.pt',weights_only=True)
    signatures=torch.load(source/f'seed{args.seed}/grad_signature.pt',weights_only=True)
    from GradientCalibratedSID.run import file_hash
    assert signatures['hard_checkpoint_sha256']==file_hash(cp)
    hard=base.item_encoder.hard_sid_table[1:].cpu().numpy();g=signatures['grad_signature'].numpy();counts=signatures['grad_count'].numpy()
    frequency=np.zeros(len(hard))
    for row in data:np.add.at(frequency,np.array(row['items'][:-2],int)-1,1)
    write(root/'protocol.json',{'source_checkpoint':str(cp),'sha256':file_hash(cp),'seed':args.seed,
          'signature':'positive-only; per-example normalized, final mean not normalized; B includes zero signatures in group mean','training':config,
          'splits_per_level_per_stratum':4,'min_group_well_profiled_items':8,'min_profile_observations_for_selection':3,
          'selection_uses_test':False,'note':'Single-seed exploratory diagnostic. Random controls match child counts; high-alignment parents approximately match occupancy.'})
    a=experiment_a(hard,g,counts,artifact['neighbor_ids'].numpy());write(root/'A_pair_conflict.json',a)
    obs=observations(base,test,args.device,len(hard));np.savez(root/'hard_test_observations.npz',**obs)
    rows=token_rows(hard,g,counts,frequency,obs)
    groups=choose_groups(rows,hard,g);write(root/'selected_groups.json',groups)
    report(root,rows,a,{})
    # Atomic ID = same backbone and item LN with a single per-item embedding table.
    atomic_dir=root/'training/atomic_id'
    torch.manual_seed(args.seed);random.seed(args.seed)
    atomic=GCSS(torch.arange(len(hard)+1).unsqueeze(1),len(hard),**base.config)
    fit(atomic,train,valid,test,config,args.seed,atomic_dir,1,'atomic_id',source={'initialization':'random, matched seed'})
    atomic,_=load_model(atomic_dir/'hard_sid_checkpoint.pt',args.device)
    atomic_obs=observations(atomic,test,args.device,len(hard));np.savez(root/'atomic_test_observations.npz',**atomic_obs);del atomic
    for r in rows:
        members=np.where(hard[:,r['level']-1]==r['token'])[0];sel=np.isin(obs['target']-1,members)
        r['atomic_ndcg']=float(atomic_obs['ndcg'][sel].mean()) if sel.any() else None
        r['hard_minus_atomic_ndcg']=r['hard_ndcg']-r['atomic_ndcg'] if sel.any() else None
    arms={'no_split':(None,False),'high_conflict_gradient':('high_conflict',False),
          'high_conflict_random':('high_conflict',True),'high_alignment_gradient':('high_alignment',False),
          'high_alignment_random':('high_alignment',True)}
    observed={}
    for name,(stratum,randomized) in arms.items():
        model,details=split_model(base,groups[stratum],g,randomized,args.seed+9000) if stratum else (copy.deepcopy(base),[])
        out=root/'training'/name;out.mkdir(parents=True,exist_ok=True);write(out/'split.json',details)
        fit(model,train,valid,test,config,args.seed,out,2,name,source={'checkpoint':str(cp),'sha256':file_hash(cp),'initial_item_vectors_identical':True})
        model,_=load_model(out/'best.pt',args.device)
        observed[name]=observations(model,test,args.device,len(hard))
        saved=json.loads((out/'result.json').read_text())['test']['NDCG@10']
        if abs(float(observed[name]['ndcg'].mean())-saved)>1e-7:
            raise RuntimeError(f'Observation/ranking mismatch for {name}')
        np.savez(out/'test_observations.npz',**observed[name]);del model
        report(root,rows,a,observed,atomic_obs)
    comparisons={}
    for stratum in groups:
        affected=np.zeros(len(hard),bool)
        for r in groups[stratum]:affected|=hard[:,r['level']-1]==r['token']
        for control in (stratum+'_random','no_split'):
            comparisons[stratum+'_gradient_minus_'+control]={}
            for subset,mask in [('all',np.ones(len(obs['target']),bool)),('affected',affected[obs['target']-1]),('unaffected',~affected[obs['target']-1])]:
                comparisons[stratum+'_gradient_minus_'+control][subset]=paired_delta(observed[stratum+'_gradient'],observed[control],mask)
    write(root/'C_comparisons.json',comparisons)
    report(root,rows,a,observed,atomic_obs)
    write(root/'completion.json',{'complete':True})


if __name__=='__main__':main()
