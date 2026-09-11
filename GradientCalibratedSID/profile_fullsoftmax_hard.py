"""Profile positive-target gradients from the full-softmax Hard SID checkpoint."""
import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from LoCoRec.locorec.data import NextItemDataset
from GradientCalibratedSID.model import load_model
from GradientCalibratedSID.profiling import profile_fullsoftmax
from GradientCalibratedSID.run import file_hash


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',default='runs/office/fullsoftmax_baselines_20260910/hard_sid/best.pt')
    p.add_argument('--dataset-dir',default='runs/office')
    p.add_argument('--output',default='runs/office/fullsoftmax_baselines_20260910/hard_sid/grad_signature.pt')
    p.add_argument('--device',default='cuda')
    p.add_argument('--threads',type=int,default=2)
    a=p.parse_args();torch.set_num_threads(a.threads)
    checkpoint=Path(a.checkpoint); output=Path(a.output)
    model,payload=load_model(checkpoint,a.device)
    config=payload['training_config'].copy();config['device']=a.device
    rows=json.loads((Path(a.dataset_dir)/'sequences.json').read_text())
    train=NextItemDataset(rows,config['max_len'],'train')
    model.eval().requires_grad_(False)
    # Full-catalog CE is used here so the diagnostic has the same objective as
    # the full-softmax baseline; negatives from the sampled training loader are
    # intentionally not used.
    generator=torch.Generator().manual_seed(payload.get('seed',2026)+8001)
    loader=DataLoader(train,batch_size=config['batch_size'],shuffle=False,generator=generator,
                      collate_fn=lambda batch:(torch.stack([x[0] for x in batch]),torch.tensor([x[1] for x in batch])))
    result=profile_fullsoftmax(model,loader,a.device)
    result['hard_checkpoint_sha256']=file_hash(checkpoint)
    result['profile_protocol']='positive target only; full-catalog CE; per-example unit direction mean; final signature not normalized'
    output.parent.mkdir(parents=True,exist_ok=True);torch.save(result,output)
    diag={'shape':list(result['grad_signature'].shape),'examples':result['examples'],'valid_examples':result['valid_examples'],'profiled_items':int(result['grad_count'].gt(0).sum()),'zero_count_items':int(result['grad_count'].eq(0).sum()),'source_checkpoint':str(checkpoint),'source_sha256':result['hard_checkpoint_sha256'],'outer_normalization':False}
    (output.parent/'profile_diagnostics.json').write_text(json.dumps(diag,indent=2))
    print(json.dumps(diag,indent=2))


if __name__=='__main__':main()
