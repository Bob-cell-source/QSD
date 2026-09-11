from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from LoCoRecSimple.model import SimpleSharing
from LoCoRecSimple.gradient_conflict_diagnostic import target_gradients, behavior_matrices, cosines


def test_target_conditioned_gradient_is_true_loss_gradient_and_positive_path_is_local():
    hard = torch.tensor([[0,0], [1,3], [1,4], [2,3], [2,4]])
    model = SimpleSharing('hard', hard, 4, None, None, 8, 4, 2, 2, 0).eval()
    sequences = torch.tensor([[0,0,1,2], [0,0,0,4]])
    candidates = torch.tensor([[3,4], [3,2]])
    full, positive, _ = target_gradients(model, sequences, candidates, 8**.5)
    loss = torch.nn.functional.cross_entropy(model(sequences, candidates)['score']/8**.5, torch.zeros(2,dtype=torch.long))
    expected = torch.autograd.grad(loss, model.item_encoder.shared_embedding.weight)[0]
    torch.testing.assert_close(full, expected)
    individual = [target_gradients(model, sequences[k:k+1], candidates[k:k+1], 8**.5)[0] for k in range(2)]
    torch.testing.assert_close(full, torch.stack(individual).mean(0), atol=1e-6, rtol=1e-5)
    assert positive[[0,1,4]].eq(0).all()
    assert positive[[2,3]].abs().sum() > 0
    assert not torch.allclose(full, positive)


def test_behavior_uses_only_training_and_cosines_detect_opposition():
    rows = [{'items':[1,2,3,4]}, {'items':[2,1,4,3]}]
    users, context, frequency = behavior_matrices(rows, 4)
    assert users[1,2] > .99
    assert users[3:].sum() == 0 and context[3:].sum() == 0
    assert frequency.tolist() == [0,2,2,0,0]
    matrix = cosines(np.array([[1.,0.],[-1.,0.],[0.,1.]]))
    np.testing.assert_array_equal(matrix, np.array([[1.,-1.,0.],[-1.,1.,0.],[0.,0.,1.]]))
