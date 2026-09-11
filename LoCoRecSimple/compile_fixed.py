"""Exact checkpoint reparameterization for a constant residual scale.

This preserves a trained model at export time, not its future training trajectory.
The assignment module is retained; the export is not a cached full item-vector table.
"""
import argparse
from copy import deepcopy
from pathlib import Path
import torch
from torch import nn
from .model import SimpleSharing, SharingItemEncoder
from LoCoRec.locorec.model import LoCoRec


class CompiledItemEncoder(SharingItemEncoder):
    def __init__(self, hard, num_tokens, soft, prior, dim):
        super().__init__('soft', hard, num_tokens, soft, prior, dim)
        self.register_buffer('private_scale',torch.zeros(len(hard)))
        self.common_bias=nn.Parameter(torch.zeros(dim))

    def forward(self,items):
        weights=self.assignment(items)
        shared=(self.shared_embedding(self.soft_sid_table[items])*weights[...,None]).sum(-2).mean(-2)
        value=self.common_bias+shared+self.private_scale[items,None]*self.private_embedding(items)
        return {'vectors':self.output_norm(value)*items.ne(0).unsqueeze(-1)}


def empty_compiled(hard,soft,prior,num_tokens,args):
    model=SimpleSharing('soft',hard,num_tokens,soft,prior,dim=args['dim'],max_len=args['max_len'],
                        num_heads=2,num_layers=2,dropout=args['dropout'])
    model.item_encoder=CompiledItemEncoder(hard,num_tokens,soft,prior,args['dim'])
    return model


@torch.no_grad()
def configure_training_ablation(model, variant):
    """Ablate the consolidated architecture after matching its random start.

    All surviving tensors retain their values. This is not checkpoint warm-starting.
    """
    enc = model.item_encoder
    if variant == 'consolidated_hard':
        enc.soft_sid_table = enc.hard_sid_table.unsqueeze(-1).clone()
        enc.candidate_prior = enc.soft_sid_table.ne(0).float()
        # Singleton softmax is exactly Hard SID; unused selector tensors are frozen.
        for name in ('selector_embedding', 'selector_query', 'selector_key'):
            getattr(enc, name).requires_grad_(False)
        enc.prior_beta_raw.requires_grad_(False)
    elif variant == 'consolidated_no_bias':
        enc.common_bias.zero_()
        enc.common_bias.requires_grad_(False)
    elif variant == 'consolidated_uniform':
        # Retain the zero-training-evidence guard and average private amplitude;
        # only remove differences among items with positive private coefficients.
        active = enc.private_scale.gt(0)
        if active.any():
            enc.private_scale[active] = enc.private_scale[active].mean()
    elif variant != 'consolidated':
        raise ValueError(variant)
    return model


@torch.no_grad()
def compile_model(original,args):
    enc=original.item_encoder
    hard=enc.soft_sid_table[:,:,0]
    n=len(hard)-1
    ids=torch.arange(n+1,device=hard.device)
    weights,_,_=enc.fusion_weights(ids)
    scale=weights[1:,1:].sum(-1)
    if not torch.allclose(scale,scale[:1].expand_as(scale),atol=1e-6,rtol=1e-6):
        raise ValueError('Residual scale is item-dependent; this export is not applicable')
    if not torch.allclose(weights[1:,0],torch.ones_like(weights[1:,0])):
        raise ValueError('Basis scale must be one')
    if not enc.candidate_prior[1:].sum(-1).gt(0).all():
        raise ValueError('All nonpadding SID levels must have at least one candidate')
    model=empty_compiled(hard,enc.soft_sid_table,enc.candidate_prior,
                         enc.shared_residual_embedding.num_embeddings-1,args).to(hard.device)
    model.sequence_encoder=deepcopy(original.sequence_encoder)
    dst=model.item_encoder
    for name in ('selector_embedding','selector_query','selector_key','output_norm'):
        getattr(dst,name).load_state_dict(getattr(enc,name).state_dict())
    dst.prior_beta_raw.copy_(enc.prior_beta_raw)
    dst.common_bias.copy_(enc.basis_projection.bias)
    dst.private_scale.copy_(weights[:,2])
    dst.shared_embedding.weight.copy_(
        enc.semantic_basis_embedding.weight@enc.basis_projection.weight.T
        +scale[0]*enc.shared_residual_embedding.weight)
    for items in ids.split(256):
        assignment,_=enc.candidate_weights(items)
        shared=enc.semantic_pool(enc.shared_residual_embedding,items,assignment)
        dst.private_embedding.weight[items]=enc.private_residual_embedding(items)-shared
    return model.eval()


def load_original(path):
    payload=torch.load(path,map_location='cpu',weights_only=True)
    state,args=payload['model'],payload['args']
    prefix='item_encoder.'
    model=LoCoRec(
        num_items=len(state[prefix+'item_frequency'])-1,
        num_semantic_tokens=len(state[prefix+'shared_residual_embedding.weight'])-1,
        soft_sid_table=state[prefix+'soft_sid_table'],candidate_prior=state[prefix+'candidate_prior'],
        local_consistency=state[prefix+'local_consistency'],item_frequency=state[prefix+'item_frequency'],
        dim=args['dim'],max_len=args['max_len'],num_heads=2,num_layers=2,dropout=args['dropout'])
    model.item_encoder.dropout=nn.Identity()
    model.load_state_dict(state)
    return model.eval(),args


def main():
    import json
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--output-dir',required=True)
    p.add_argument('--verify-test',action='store_true')
    args=p.parse_args()
    torch.set_num_threads(2)
    original,settings=load_original(args.checkpoint)
    compiled=compile_model(original,settings)
    errors=[]
    with torch.no_grad():
        for ids in torch.arange(1,len(compiled.item_encoder.hard_sid_table)).split(256):
            errors.append((original.item_encoder(ids)['vectors']-compiled.item_encoder(ids)['vectors']).abs().max().item())
    result={'source':args.checkpoint,'max_catalog_vector_error':max(errors),
            'original_parameters':sum(p.numel() for p in original.parameters()),
            'compiled_parameters':sum(p.numel() for p in compiled.parameters()),
            'note':'Checkpoint equivalence only; future optimization is not equivalent. Assignment module retained.'}
    if result['max_catalog_vector_error']>1e-5:raise ValueError(result)
    if args.verify_test:
        from torch.utils.data import DataLoader
        from LoCoRec.locorec.data import NextItemDataset,collate_eval
        from .experiment import evaluate
        rows=json.loads((Path(settings['dataset_dir'])/'sequences.json').read_text())
        loader=DataLoader(NextItemDataset(rows,settings['max_len'],'test'),
                          batch_size=settings['eval_batch_size'],collate_fn=collate_eval)
        metrics,per_user=evaluate(compiled,loader,'cpu',len(compiled.item_encoder.hard_sid_table)-1)
        reference=torch.load(Path(args.checkpoint).parent/'test_per_user.pt',weights_only=True)
        historical_difference=(per_user-reference).abs()
        result['historical_reference_changed_users']=int(historical_difference.gt(1e-7).sum())
        result['historical_reference_max_ndcg10_difference']=float(historical_difference.max())
        if historical_difference.gt(1e-7).any():
            # The saved reference may have been evaluated on CUDA. Re-evaluate
            # the unmodified original on the same CPU backend before attributing
            # a near-boundary ranking change to compilation.
            original_metrics,reference=evaluate(original,loader,'cpu',len(compiled.item_encoder.hard_sid_table)-1)
            result['same_backend_original_test']=original_metrics
        torch.testing.assert_close(per_user,reference,atol=1e-7,rtol=1e-7)
        result.update(test=metrics,max_per_user_ndcg10_difference=float((per_user-reference).abs().max()))
    output=Path(args.output_dir);output.mkdir(parents=True,exist_ok=True)
    torch.save({'model':compiled.state_dict(),'args':settings,'source':args.checkpoint},output/'compiled.pt')
    (output/'report.json').write_text(json.dumps(result,indent=2))
    print(result)


if __name__=='__main__':main()
