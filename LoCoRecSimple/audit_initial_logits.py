import json,random,torch,math
from pathlib import Path
from functools import partial
from torch.utils.data import DataLoader
from LoCoRecSimple.model import SimpleSharing,initialize_matched
from LoCoRec.locorec.model import LoCoRec
from LoCoRec.locorec.soft_sid import build_semantic_table,build_soft_sid_table,SoftSIDConfig,build_train_item_frequency
from LoCoRec.locorec.data import NextItemDataset,collate_train
from CCSR.trainer import NegativeSampler

torch.set_num_threads(1)
root=Path('runs/office');rows=json.loads((root/'sequences.json').read_text());n=json.loads((root/'stats.json').read_text())['num_items']
hard,codes,tokens=build_semantic_table(json.loads((root/'semantic_ids_rq.json').read_text()),n)
soft,prior,rel=build_soft_sid_table(hard,codes,SoftSIDConfig(top_m=4,min_overlap_slots=3,leave_one_level_out=False,tie_break_seed=2026))
freq=build_train_item_frequency(rows,n)
ds=NextItemDataset(rows,50,'train')
random.seed(2026001)
seq,cand=next(iter(DataLoader(ds,batch_size=256,shuffle=True,generator=torch.Generator().manual_seed(2026),collate_fn=partial(collate_train,sampler=NegativeSampler(n,100)))))
results=[]
for variant in ['id','hard','soft','full','full_zero_projection_bias']:
 torch.manual_seed(2026)
 kw=dict(dim=128,max_len=50,num_heads=2,num_layers=2,dropout=.2)
 if variant.startswith('full'):
  model=LoCoRec(n,tokens,soft,prior,rel,freq,**kw);model.item_encoder.dropout=torch.nn.Identity()
 else:model=SimpleSharing(variant,hard,tokens,soft,prior,**kw)
 initialize_matched(model,2026,128,50,2,2,.2,n,tokens)
 model.eval()
 with torch.no_grad():
  if variant.endswith('bias'):model.item_encoder.basis_projection.bias.zero_()
  vec=model.item_encoder(torch.arange(1,n+1))['vectors']
  sim= torch.nn.functional.normalize(vec,dim=-1)
  mean_cos=(sim.sum(0).square().sum()-n)/(n*(n-1))
  logits=model(seq,cand)['score'];centered=logits-logits.mean(-1,keepdim=True)
  result={'variant':variant,'CE':torch.nn.functional.cross_entropy(logits,torch.zeros(256,dtype=torch.long)).item(),
   'row_centered_logit_std':centered.std(-1).mean().item(),'mean_pair_item_cosine':mean_cos.item(),
   'CE_div_sqrt_d':torch.nn.functional.cross_entropy(logits/math.sqrt(128),torch.zeros(256,dtype=torch.long)).item()}
  results.append(result);print(result)
Path('runs/office/initial_logit_audit_20260910.json').write_text(json.dumps(results,indent=2))
