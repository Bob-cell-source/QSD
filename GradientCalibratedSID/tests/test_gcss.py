import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from GradientCalibratedSID.model import GCSS, load_model
from GradientCalibratedSID.candidates import build_candidates
from GradientCalibratedSID.calibration import compatibility, sharing_weights, shuffled_compatibility
from GradientCalibratedSID.profiling import positive_gradients, aggregate_directions


@pytest.fixture
def model():
    torch.set_num_threads(1)
    torch.manual_seed(17)
    return GCSS(torch.tensor([[0,0],[1,4],[1,5],[2,4],[2,5]]), 5,
                dim=8, max_len=3, heads=2, layers=1, dropout=0.).eval()


def test_exact_hard_and_fixed_lookup(model):
    ids = torch.tensor([0,1,2,3])
    encoder = model.item_encoder
    expected = encoder.output_norm(encoder.sid_embedding(encoder.hard_sid_table[ids]).mean(-2))
    expected[0] = 0
    torch.testing.assert_close(model.encode_items(ids), expected)
    before = model.encode_items(ids).clone()
    encoder.set_lookup(encoder.hard_sid_table.unsqueeze(-1), encoder.sharing_weight)
    torch.testing.assert_close(model.encode_items(ids), before)
    assert not encoder.sharing_weight.requires_grad
    assert all(not any(s in n for s in ('private','gate','attention_query')) for n,_ in encoder.named_parameters())


def test_candidates_lexicographic_and_support():
    hard = np.array([[1,4],[1,5],[1,6],[2,4],[3,7]])
    text = np.array([[1.,0.],[0.,1.],[.9,.1],[-1.,0.],[0.,1.]])
    a = build_candidates(hard,text,max_neighbors=2,max_candidates=2)
    assert a['neighbor_ids'][0].tolist() == [3,2]
    assert a['neighbor_ids'][4].tolist() == [0,0]
    assert a['candidate_ids'][0,1,0] == 4
    # Three distinct tokens have support 1/3; truncation must not change q.
    torch.testing.assert_close(a['candidate_support'][0,1], torch.tensor([1/3,1/3]))
    assert a['candidate_mask'][4].sum() == 2
    assert a['candidate_support'][4,:,0].eq(1).all()
    assert all(i+1 not in a['neighbor_ids'][i].tolist() for i in range(5))
    b = build_candidates(hard,text,max_neighbors=0,max_candidates=1)
    assert b['candidate_ids'].squeeze(-1).equal(torch.tensor(hard))


def test_positive_gradient_isolation_and_batch_equivalence(model):
    model.requires_grad_(False)
    history = torch.tensor([[0,1,2],[0,2,1]])
    targets = torch.tensor([1,2])
    # Each positive is also the OTHER row's negative and occurs in history.
    negatives = torch.tensor([[2,3],[1,4]])
    before = {k:v.clone() for k,v in model.state_dict().items()}
    batch = positive_gradients(model,history,targets,negatives)
    single = torch.cat([positive_gradients(model,history[i:i+1],targets[i:i+1],negatives[i:i+1]) for i in range(2)])
    torch.testing.assert_close(batch,single,atol=1e-6,rtol=1e-5)
    assert batch.norm() > 0
    assert all(p.grad is None for p in model.parameters())
    assert all(v.equal(before[k]) for k,v in model.state_dict().items())
    # Independent reference with separate per-level leaves verifies 1/L.
    with torch.no_grad():
        h,_ = model.encode_sequence(history)
        neg = model.encode_items(negatives)
    levels = model.item_encoder.sid_embedding(model.item_encoder.hard_sid_table[targets]).detach().requires_grad_()
    pos = model.item_encoder.output_norm(levels.mean(1))
    logits = torch.cat([(h*pos).sum(-1,keepdim=True),(h[:,None]*neg).sum(-1)],-1) / 8**.5
    loss = torch.nn.functional.cross_entropy(logits,torch.zeros(2,dtype=torch.long),reduction='sum')
    level_grad, = torch.autograd.grad(loss,levels)
    torch.testing.assert_close(level_grad,batch[:,None].expand(-1,2,-1)/2)


def test_directional_mean_preserves_cancellation():
    sums = torch.zeros(4,2,dtype=torch.double)
    counts = torch.zeros(4,dtype=torch.long)
    aggregate_directions(sums,counts,torch.tensor([1,1,2,2,3]),
                         torch.tensor([[2.,0.],[-3.,0.],[1.,0.],[0.,1.],[0.,0.]]))
    mean = sums/counts.clamp_min(1)[:,None]
    torch.testing.assert_close(mean[1],torch.zeros(2,dtype=torch.double),atol=1e-10,rtol=0)
    torch.testing.assert_close(mean[2],torch.tensor([.5,.5],dtype=torch.double))
    assert counts.tolist() == [0,2,2,0]
    with pytest.raises(ValueError,match='Non-finite'):
        aggregate_directions(sums,counts,torch.tensor([1]),torch.tensor([[float('nan'),0.]]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA precision regression')
def test_cuda_profiling_restores_precision(model):
    model = model.cuda().requires_grad_(False)
    h = torch.tensor([[0,1,2],[0,2,1]],device='cuda')
    t = torch.tensor([1,2],device='cuda')
    n = torch.tensor([[2,3],[1,4]],device='cuda')
    before = (torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
    batch = positive_gradients(model,h,t,n)
    singles = torch.cat([positive_gradients(model,h[i:i+1],t[i:i+1],n[i:i+1]) for i in range(2)])
    torch.testing.assert_close(batch,singles,atol=1e-5,rtol=1e-4)
    assert before == (torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)


def test_compatibility_no_self_no_normalization():
    a = build_candidates(np.array([[1,4],[1,5],[2,4]]),np.eye(3),max_neighbors=2,max_candidates=3)
    sig = torch.tensor([[.5,0.],[.2,0.],[-.4,0.]])
    c = compatibility(a,sig)
    assert c[0,0,0].item() == pytest.approx(.1)
    assert c[0,1,0].item() == pytest.approx(-.2)
    assert c[1,1,0].item() == 0  # No neighbor shares original token 5.
    zero = compatibility(a,torch.zeros_like(sig))
    assert zero.eq(0).all()
    w = sharing_weights(a['candidate_support'],a['candidate_mask'],c,0.)
    expected = a['candidate_support']/a['candidate_support'].sum(-1,keepdim=True)
    torch.testing.assert_close(w,expected)
    calibrated = sharing_weights(a['candidate_support'],a['candidate_mask'],c,2.)
    assert calibrated[0,0,0] > w[0,0,0]
    assert calibrated[~a['candidate_mask']].eq(0).all()
    shuffled = shuffled_compatibility(c,a['candidate_mask'],33)
    for i in range(3):
        for l in range(2):
            mask = a['candidate_mask'][i,l]
            assert c[i,l,mask].sort().values.equal(shuffled[i,l,mask].sort().values)


def test_reject_old_checkpoint(tmp_path):
    path = tmp_path/'old.pt'
    torch.save({'model':{}},path)
    with pytest.raises(ValueError,match='pure GCSS'):
        load_model(path)


def test_end_to_end(tmp_path):
    n = 24
    data = tmp_path/'data'
    data.mkdir()
    rows = [{'user_id':i,'items':[(i+j)%n+1 for j in range(7)]} for i in range(n)]
    (data/'sequences.json').write_text(json.dumps(rows))
    (data/'stats.json').write_text(json.dumps({'num_items':n}))
    sid = {'codebook_sizes':[4,4], 'semantic_ids':{str(i+1):[i%4,(i//4)%4] for i in range(n)}}
    (data/'sid.json').write_text(json.dumps(sid))
    np.save(data/'item_text_embeddings.npy',np.random.default_rng(1).normal(size=(n,5)))
    (data/'embedding_item_ids.json').write_text(json.dumps(list(range(1,n+1))))
    output = tmp_path/'output'
    args = [sys.executable,'-m','GradientCalibratedSID.run','--dataset-dir',str(data),
            '--semantic-ids',str(data/'sid.json'),'--output-dir',str(output),'--device','cpu',
            '--dim','8','--max-len','4','--layers','1','--batch-size','8','--negatives','2',
            '--epochs','2','--patience','1','--H','3','--M','2','--threads','1']
    result = subprocess.run(args,capture_output=True,text=True,timeout=90)
    assert result.returncode == 0, result.stdout+'\n'+result.stderr
    assert json.loads((output/'completion.json').read_text())['complete']
    seed = output/'seed2026'
    payload = torch.load(seed/'grad_signature.pt',weights_only=True)
    assert payload['grad_signature'].shape == (n,8)
    assert json.loads((seed/'profile_debug.json').read_text())['frozen_state_verified']
    params = set()
    for variant in ('hard','semantic_soft','gradient_calibrated','random_compatibility'):
        path = seed/'stage2'/variant
        report = json.loads((path/'result.json').read_text())
        assert report['fixed_lookup_verified']
        params.add(report['parameters'])
        loaded,_ = load_model(path/'best.pt')
        assert loaded.encode_items(torch.tensor([1,2])).shape == (2,8)
    assert len(params) == 1
