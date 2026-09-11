import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from LoCoRec.locorec.model import LoCoRec
from LoCoRecSimple.compile_fixed import compile_model
from LoCoRecSimple.compile_fixed import configure_training_ablation
from copy import deepcopy


def test_compiled_fixed_fusion_preserves_vectors_and_scores():
    hard=torch.tensor([[0,0],[1,3],[1,4],[2,3],[2,4]])
    soft=torch.stack([hard,hard.flip(0)],-1)
    prior=soft.ne(0).float();prior/=prior.sum(-1,keepdim=True).clamp_min(1)
    original=LoCoRec(4,4,soft,prior,torch.tensor([0.,.2,.4,.8,1.]),
                     torch.tensor([0.,0.,3.,10.,50.]),dim=8,max_len=4,num_heads=2,num_layers=2,dropout=0).eval()
    # Padding has no active candidates in the real preprocessing.
    original.item_encoder.soft_sid_table[0].zero_()
    original.item_encoder.candidate_prior[0].zero_()
    compiled=compile_model(original,{'dim':8,'max_len':4,'dropout':0.})
    ids=torch.arange(5)
    torch.testing.assert_close(original.item_encoder(ids)['vectors'],compiled.item_encoder(ids)['vectors'])
    seq=torch.tensor([[0,0,1,2],[0,0,0,3]]);cand=torch.tensor([[3,4],[2,1]])
    torch.testing.assert_close(original(seq,cand)['score'],compiled(seq,cand)['score'])
    assert not hasattr(compiled.item_encoder,'basis_projection')
    assert not hasattr(compiled.item_encoder,'residual_gate')
    for variant in ('consolidated', 'consolidated_hard', 'consolidated_no_bias', 'consolidated_uniform'):
        ablated = configure_training_ablation(deepcopy(compiled), variant)
        enc = ablated.item_encoder
        torch.testing.assert_close(enc.shared_embedding.weight, compiled.item_encoder.shared_embedding.weight)
        torch.testing.assert_close(enc.private_embedding.weight, compiled.item_encoder.private_embedding.weight)
        if variant == 'consolidated_hard':
            expected = enc.output_norm(enc.common_bias + enc.shared_embedding(enc.hard_sid_table[ids]).mean(-2)
                                      + enc.private_scale[ids, None] * enc.private_embedding(ids))
            torch.testing.assert_close(enc(ids)['vectors'][1:], expected[1:])
        if variant == 'consolidated_uniform':
            assert enc.private_scale[1] == 0  # Cold item guard survives the ablation.
            torch.testing.assert_close(enc.private_scale[2:], enc.private_scale[2].expand(3))
        score = ablated(seq, cand)['score']
        torch.nn.functional.cross_entropy(score, torch.zeros(2, dtype=torch.long)).backward()
        assert enc.shared_embedding.weight.grad is not None
        assert enc.private_embedding.weight.grad is not None
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in ablated.parameters())
