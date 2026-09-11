"""Full-softmax token-splitting intervention from the matched Hard checkpoint."""
import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from LoCoRec.locorec.data import NextItemDataset, collate_eval
from GradientCalibratedSID.model import load_model
from GradientCalibratedSID.diagnose_abc import token_rows, choose_groups, split_model, observations
from GradientCalibratedSID.fullsoftmax_baselines import fit, collate_full


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',default='runs/office/fullsoftmax_baselines_20260910/hard_sid/best.pt');p.add_argument('--dataset-dir',default='runs/office');p.add_argument('--signature',default='runs/office/fullsoftmax_baselines_20260910/hard_sid/grad_signature.pt');p.add_argument('--observations',default='runs/office/fullsoftmax_baselines_20260910/hard_sid/test_observations.npz');p.add_argument('--output',default='runs/office/fullsoftmax_baselines_20260910/intervention');p.add_argument('--device',default='cuda');p.add_argument('--seed',type=int,default=2026);p.add_argument('--epochs',type=int,default=30);p.add_argument('--patience',type=int,default=10)
    a=p.parse_args();torch.set_num_threads(2);root=Path(a.output);root.mkdir(parents=True,exist_ok=True);base,payload=load_model(a.checkpoint,a.device);base.eval()
    cfg=payload['training_config'].copy();cfg.update(device=a.device,epochs=a.epochs,patience=a.patience)
    rows=json.loads((Path(a.dataset_dir)/'sequences.json').read_text());train=NextItemDataset(rows,cfg['max_len'],'train');valid=DataLoader(NextItemDataset(rows,cfg['max_len'],'valid'),batch_size=cfg['batch_size'],collate_fn=collate_eval);test=DataLoader(NextItemDataset(rows,cfg['max_len'],'test'),batch_size=cfg['batch_size'],collate_fn=collate_eval)
    sig=torch.load(a.signature,map_location='cpu',weights_only=True);g=sig['grad_signature'].numpy();counts=sig['grad_count'].numpy();hard=base.item_encoder.hard_sid_table[1:].cpu().numpy();obs=observations(base,test,a.device,len(hard));frequency=np.zeros(len(hard));
    for row in rows:np.add.at(frequency,np.asarray(row['items'][:-2],int)-1,1)
    token_data=token_rows(hard,g,counts,frequency,obs);groups=choose_groups(token_data,hard,g);(root/'selected_groups.json').write_text(json.dumps(groups,indent=2))
    # All arms have identical optimizer/training settings. No-split is a
    # continuation control from exactly the same full-softmax checkpoint.
    arms={'no_split':(None,False),'low_C_gradient':('high_conflict',False),'low_C_random':('high_conflict',True),'high_C_gradient':('high_alignment',False),'high_C_random':('high_alignment',True)}
    results={}
    for name,(group_name,randomized) in arms.items():
        if group_name is None:model=copy.deepcopy(base);details=[]
        else:model,details=split_model(base,groups[group_name],g,randomized,a.seed+7000)
        out=root/name;out.mkdir(parents=True,exist_ok=True);(out/'split.json').write_text(json.dumps(details,indent=2))
        results[name]=fit(name,model,train,valid,test,cfg,a.seed,out)
    (root/'summary.json').write_text(json.dumps(results,indent=2));lines=['# Full-softmax token split intervention','', '| Arm | Test NDCG@10 | Best epoch | Parameters |','|---|---:|---:|---:|']
    for name,r in results.items():lines.append(f"| {name} | {r['test']['NDCG@10']:.6f} | {r['best_epoch']} | {r['parameters']} |")
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
