import numpy as np
import torch
from GradientCalibratedSID.model import GCSS
from GradientCalibratedSID.diagnose_abc import cluster, split_model, token_rows, pair_stats


def test_pair_and_group_mean_include_zero_signature():
    hard=np.array([[1],[1],[1]])
    g=np.array([[1.,0.],[-.5,0.],[0.,0.]])
    obs={'target':np.array([1,2,3]),'recall':np.array([1.,0.,0.]),
         'ndcg':np.array([1.,0.,0.]),'loss':np.array([1.,2.,3.])}
    rows=token_rows(hard,g,np.array([3,3,0]),np.ones(3),obs)
    assert abs(rows[0]['mean_compatibility']-(-1/6))<1e-10
    pairs=np.array([[0,1],[0,2],[1,2]])
    assert pair_stats(g,pairs)['negative_fraction']==1/3


def test_cluster_split_preserves_predictions_and_random_sizes():
    torch.set_num_threads(1)
    table=torch.tensor([[0,0],[1,3],[1,4],[1,3],[1,4]])
    model=GCSS(table,4,dim=8,max_len=3,layers=1,heads=2,dropout=0.).eval()
    g=np.array([[1.,0.],[.8,0.],[-1.,0.],[-.8,0.]])
    labels=cluster(np.arange(4),g)
    assert labels[0]==labels[1] and labels[2]==labels[3] and labels[0]!=labels[2]
    groups=[{'level':1,'token':1}]
    guided,details=split_model(model,groups,g)
    control,random_details=split_model(model,groups,g,True,12)
    assert details[0]['child_sizes']==random_details[0]['child_sizes']==[2,2]
    h=torch.tensor([[0,1,2],[0,3,4]])
    cand=torch.tensor([[1,2,3,4],[1,2,3,4]])
    with torch.no_grad():
        torch.testing.assert_close(model(h,cand)['score'],guided(h,cand)['score'],atol=0,rtol=0)
        torch.testing.assert_close(model(h,cand)['score'],control(h,cand)['score'],atol=0,rtol=0)
    assert guided.item_encoder.sid_embedding.num_embeddings==6
    assert sum(p.numel() for p in guided.parameters())==sum(p.numel() for p in control.parameters())
    assert model.item_encoder.hard_sid_table.equal(table)


def test_observation_metrics_match_training_evaluator_with_sid_collisions():
    from torch.utils.data import DataLoader
    from LoCoRec.locorec.data import NextItemDataset,collate_eval
    from LoCoRecSimple.experiment import evaluate
    from GradientCalibratedSID.diagnose_abc import observations
    table=torch.tensor([[0,0]]+[[1,3]]*24)
    model=GCSS(table,4,dim=8,max_len=3,layers=1,heads=2,dropout=0.).eval()
    rows=[{'items':[1,2,3,target]} for target in range(1,25)]
    loader=DataLoader(NextItemDataset(rows,3,'test'),batch_size=2,collate_fn=collate_eval)
    metrics,per_user=evaluate(model,loader,'cpu',24)
    obs=observations(model,loader,'cpu',24)
    np.testing.assert_array_equal(obs['ndcg'],per_user.numpy())
    assert abs(obs['ndcg'].mean()-metrics['NDCG@10'])<1e-7
