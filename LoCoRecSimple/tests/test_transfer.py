import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from LoCoRecSimple.transfer import SIDGradientTransfer


def test_transfer_operator_matches_dense_projectors_and_is_psd():
    hard=torch.tensor([[0,0],[1,4],[1,5],[2,5],[3,6]])
    gradient=torch.randn(5,7,dtype=torch.double)
    c=torch.zeros(4,4,dtype=torch.double)
    for level in range(2):
        equal=hard[1:,level,None].eq(hard[None,1:,level]).double()
        c+=equal/equal.sum(-1,keepdim=True)/2
    for eta in (0.,.5,1.):
        k=torch.eye(4,dtype=torch.double)+eta*(c-torch.diag(c.diagonal()))
        result=SIDGradientTransfer(hard,eta)(gradient)
        torch.testing.assert_close(result[1:],k@gradient[1:])
        torch.testing.assert_close(result[0],gradient[0])
        torch.testing.assert_close(k.diagonal(),torch.ones(4,dtype=torch.double))
        eig=torch.linalg.eigvalsh(k)
        assert eig.min() >= 1-eta-1e-12 and eig.max() <= 1+eta+1e-12
        # Unique SID tokens give no other-item transfer.
        torch.testing.assert_close(result[4],gradient[4])


def test_registered_gradient_hook_changes_backward_not_forward():
    hard=torch.tensor([[0,0],[1,3],[1,4],[2,4]])
    p=torch.nn.Parameter(torch.randn(4,3,dtype=torch.double))
    original=p.detach().clone()
    operator=SIDGradientTransfer(hard)
    handle=p.register_hook(operator)
    p.square().sum().backward()
    torch.testing.assert_close(p,original)
    torch.testing.assert_close(p.grad,operator(2*original))
    handle.remove()
