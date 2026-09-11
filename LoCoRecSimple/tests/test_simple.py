import sys
from pathlib import Path
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from LoCoRecSimple.model import SimpleSharing, initialize_matched
from LoCoRecSimple.experiment import evaluate
from LoCoRec.locorec.data import NextItemDataset, collate_eval
from torch.utils.data import DataLoader


def make(variant,single=False):
    hard=torch.tensor([[0,0],[1,3],[1,4],[2,3],[2,4]])
    soft=hard.unsqueeze(-1) if single else torch.stack([hard,hard.flip(0)],-1)
    prior=soft.ne(0).float();prior=prior/prior.sum(-1,keepdim=True).clamp_min(1)
    return SimpleSharing(variant,hard,4,soft,prior,8,4,2,1,0)


def test_hard_is_shared_plus_private_and_padding():
    m=make('hard');e=m.item_encoder;items=torch.tensor([0,1,2])
    expected=e.output_norm(e.private_embedding(items)+e.shared_embedding(e.hard_sid_table[items]).mean(-2))
    torch.testing.assert_close(e(items)['vectors'][1:],expected[1:])
    assert e(items)['vectors'][0].eq(0).all()
    assert not any('basis' in n or 'gate' in n or 'projection' in n for n,_ in e.named_parameters())


def test_soft_single_candidate_equals_hard_with_shared_weights():
    hard,soft=make('hard'),make('soft',True)
    soft.sequence_encoder.load_state_dict(hard.sequence_encoder.state_dict())
    for name in ('private_embedding','shared_embedding','output_norm'):
        getattr(soft.item_encoder,name).load_state_dict(getattr(hard.item_encoder,name).state_dict())
    seq=torch.tensor([[0,0,1,2],[0,0,0,3]]);cand=torch.tensor([[3,4],[2,1]])
    torch.testing.assert_close(hard(seq,cand)['score'],soft(seq,cand)['score'])


@pytest.mark.parametrize('variant',['id','hard','soft'])
def test_finite_backward(variant):
    m=make(variant,True);seq=torch.tensor([[0,0,1,2],[0,0,0,3]]);cand=torch.tensor([[3,4],[2,1]])
    score=m(seq,cand)['score']
    torch.nn.functional.cross_entropy(score,torch.zeros(2,dtype=torch.long)).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
    assert m.item_encoder.private_embedding.weight.grad is not None


def test_matching_initialization_and_catalog_masking():
    hard,soft=make('hard'),make('soft',True)
    for m in (hard,soft):initialize_matched(m,2026,8,4,2,1,0,4,4)
    torch.testing.assert_close(hard.item_encoder.private_embedding.weight,soft.item_encoder.private_embedding.weight)
    torch.testing.assert_close(hard.item_encoder.shared_embedding.weight,soft.item_encoder.shared_embedding.weight)
    rows=[{'items':[1,2,3,4]},{'items':[3,2,1,4]}]
    loader=DataLoader(NextItemDataset(rows,4,'test'),batch_size=2,collate_fn=collate_eval)
    metric,_=evaluate(hard,loader,'cpu',4,2)
    assert metric['NDCG@10']==1


@pytest.mark.parametrize('variants',[
    ['id','hard','soft','full'],
    ['full_fixed','full_global','full_equal','hard_shuffled','id_transfer','id_transfer_shuffled'],
    ['id_scaled','hard_scaled','soft_scaled','full_scaled'],
    ['consolidated_scaled','consolidated_hard_scaled','consolidated_no_bias_scaled','consolidated_uniform_scaled'],
])
def test_end_to_end_four_variants(tmp_path,variants):
    import json,subprocess
    (tmp_path/'sequences.json').write_text(json.dumps([{'items':[1,2,3,4,5]},{'items':[2,3,4,5,6]}]))
    (tmp_path/'stats.json').write_text(json.dumps({'num_items':8}))
    (tmp_path/'sid.json').write_text(json.dumps({'codebook_sizes':[2]*4,'semantic_ids':{str(i):[i%2,(i//2)%2,(i//4)%2,0] for i in range(1,9)}}))
    subprocess.run([sys.executable,'-m','LoCoRecSimple.experiment','--dataset-dir',str(tmp_path),
                    '--semantic-ids',str(tmp_path/'sid.json'),'--output-dir',str(tmp_path/'out'),
                    '--seeds','2026','--epochs','2','--gate-warmup','0','--device','cpu','--dim','8',
                    '--max-len','4','--batch-size','2','--negatives','2','--threads','1','--variants',*variants],
                   check=True,capture_output=True,cwd=Path(__file__).resolve().parents[2])
    result=json.loads((tmp_path/'out/results.json').read_text())
    assert len(result)==len(variants) and {r['variant'] for r in result}==set(variants)
    assert all((tmp_path/'out'/r['variant']/'seed2026'/'best.pt').exists() for r in result)


def test_fusion_controls_preserve_base_and_remove_item_dependence():
    from copy import deepcopy
    from LoCoRec.locorec.model import LoCoRec
    from LoCoRecSimple.model import configure_fusion_control
    hard=torch.tensor([[0,0],[1,3],[1,4],[2,3],[2,4]])
    prior=hard.ne(0).float().unsqueeze(-1)
    model=LoCoRec(4,4,hard.unsqueeze(-1),prior,torch.tensor([0.,.2,.4,.8,1.]),
                  torch.tensor([0.,1.,3.,10.,50.]),dim=8,max_len=4,num_heads=2,num_layers=1,dropout=0)
    items=torch.arange(5)
    fixed=deepcopy(model);configure_fusion_control(fixed,'full_fixed')
    torch.testing.assert_close(model.item_encoder(items)['vectors'],fixed.item_encoder(items)['vectors'])
    equal=deepcopy(model);configure_fusion_control(equal,'full_equal')
    weights,_,_=equal.item_encoder.fusion_weights(items)
    torch.testing.assert_close(weights[1:],torch.ones(4,3))
    assert weights[0].eq(0).all()
    global_model=deepcopy(model);configure_fusion_control(global_model,'full_global')
    weights,_,_=global_model.item_encoder.fusion_weights(items)
    torch.testing.assert_close(weights[1:],weights[1].expand(4,3))
    global_model.item_encoder(items)['vectors'].square().sum().backward()
    assert global_model.item_encoder.residual_gate.logits.grad is not None
