"""Train matched full-softmax Hard-SID and Atomic-ID SASRec baselines."""
import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path
from functools import partial

import torch
from torch.utils.data import DataLoader

from LoCoRec.locorec.data import NextItemDataset, collate_eval
from LoCoRecSimple.experiment import evaluate
from GradientCalibratedSID.model import GCSS
from GradientCalibratedSID.trainer import tensor_hash


def collate_full(batch):
    sequences, targets, _ = zip(*batch)
    return torch.stack(sequences), torch.tensor(targets, dtype=torch.long)


def loader(data, batch_size, seed, shuffle):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle, generator=generator,
                      collate_fn=collate_full)


def save_checkpoint(model, path, variant, config, seed, epoch, best):
    torch.save({'format':'gcss-v1', 'config':model.config, 'model':model.state_dict(),
                'stage':1, 'variant':variant, 'seed':seed, 'epoch':epoch,
                'best_valid_NDCG@10':best, 'training_config':config}, path)


def fit(variant, model, train, valid, test, config, seed, output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    result_path=output/'result.json'; checkpoint_path=output/'best.pt'
    if result_path.exists() and checkpoint_path.exists():return json.loads(result_path.read_text())
    device=config['device'];model.to(device).requires_grad_(True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
    fixed_ids=tensor_hash(model.item_encoder.candidate_ids);fixed_weight=tensor_hash(model.item_encoder.sharing_weight)
    best=-float('inf'); best_epoch=0; stale=0; history=[]; start=time.perf_counter()
    for epoch in range(1,config['epochs']+1):
        model.train(); total=0.; count=0
        for history_items, targets in loader(train,config['batch_size'],seed*1000+epoch,True):
            history_items=history_items.to(device);targets=targets.to(device)
            h,_=model.encode_sequence(history_items)
            # Full catalog is small enough for one dense matrix multiplication.
            catalog=model.encode_items(torch.arange(1,config['num_items']+1,device=device))
            logits=(h@catalog.T)/math.sqrt(model.config['dim'])
            loss=torch.nn.functional.cross_entropy(logits,targets-1)
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
            optimizer.step();total+=float(loss.detach())*len(targets);count+=len(targets)
        metrics,_=evaluate(model,valid,device,config['num_items'])
        record={'epoch':epoch,'loss':total/count,'valid':metrics,'elapsed_seconds':time.perf_counter()-start}
        history.append(record);(output/'history.json').write_text(json.dumps(history,indent=2));
        print(json.dumps({'variant':variant,**record}),flush=True)
        if metrics['NDCG@10']>best:
            best,best_epoch,stale=metrics['NDCG@10'],epoch,0
            save_checkpoint(model,checkpoint_path,variant,config,seed,best_epoch,best)
        else:
            stale+=1
            if stale>=config['patience']:break
    if fixed_ids!=tensor_hash(model.item_encoder.candidate_ids) or fixed_weight!=tensor_hash(model.item_encoder.sharing_weight):
        raise RuntimeError('Hard lookup buffers changed')
    state=torch.load(checkpoint_path,map_location=device,weights_only=True);model.load_state_dict(state['model'])
    test_metrics,per_user=evaluate(model,test,device,config['num_items']);torch.save(per_user,output/'test_per_user.pt')
    result={'variant':variant,'seed':seed,'best_epoch':best_epoch,'best_valid_NDCG@10':best,
            'test':test_metrics,'parameters':sum(p.numel() for p in model.parameters()),
            'loss':'full-softmax cross entropy over all catalog items','fixed_lookup_verified':True,
            'elapsed_seconds':time.perf_counter()-start,'training_config':config}
    result_path.write_text(json.dumps(result,indent=2));return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset-dir',default='runs/office');p.add_argument('--semantic-ids',default='runs/office/semantic_ids_rq.json');p.add_argument('--output-dir',default='runs/office/fullsoftmax_baselines_20260910');p.add_argument('--device',default='cuda');p.add_argument('--seed',type=int,default=2026);p.add_argument('--dim',type=int,default=128);p.add_argument('--max-len',type=int,default=50);p.add_argument('--heads',type=int,default=2);p.add_argument('--layers',type=int,default=2);p.add_argument('--batch-size',type=int,default=256);p.add_argument('--epochs',type=int,default=100);p.add_argument('--patience',type=int,default=10);p.add_argument('--threads',type=int,default=2);p.add_argument('--dropout',type=float,default=.2);p.add_argument('--lr',type=float,default=.001);p.add_argument('--weight-decay',type=float,default=.0001)
    a=p.parse_args();torch.set_num_threads(a.threads);random.seed(a.seed);torch.manual_seed(a.seed)
    root=Path(a.output_dir);root.mkdir(parents=True,exist_ok=True);data_dir=Path(a.dataset_dir)
    rows=json.loads((data_dir/'sequences.json').read_text());stats=json.loads((data_dir/'stats.json').read_text());sid=json.loads(Path(a.semantic_ids).read_text())
    from LoCoRec.locorec.soft_sid import build_semantic_table
    hard,_,tokens=build_semantic_table(sid,stats['num_items'])
    config=vars(a).copy();config['num_items']=stats['num_items']
    train=NextItemDataset(rows,a.max_len,'train');valid=DataLoader(NextItemDataset(rows,a.max_len,'valid'),batch_size=a.batch_size,collate_fn=collate_eval);test=DataLoader(NextItemDataset(rows,a.max_len,'test'),batch_size=a.batch_size,collate_fn=collate_eval)
    results={}
    for variant,table,num_tokens in [('hard_sid',hard,tokens),('atomic_id',torch.arange(stats['num_items']+1).unsqueeze(1),stats['num_items'])]:
        random.seed(a.seed);torch.manual_seed(a.seed)
        model=GCSS(table,num_tokens,a.dim,a.max_len,a.heads,a.layers,a.dropout)
        results[variant]=fit(variant,model,train,valid,test,config,a.seed,root/variant)
    (root/'summary.json').write_text(json.dumps(results,indent=2))
    lines=['# Full-softmax matched baselines','', '| Variant | Test NDCG@10 | Test HR@10 | Best epoch | Parameters |','|---|---:|---:|---:|---:|']
    for v,r in results.items():lines.append(f"| {v} | {r['test']['NDCG@10']:.6f} | {r['test']['HR@10']:.6f} | {r['best_epoch']} | {r['parameters']} |")
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
