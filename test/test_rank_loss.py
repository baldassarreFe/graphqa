import torch

from proteins.utils import rank_loss


def test_rank_loss():
    # same ordering, zero loss
    true = torch.tensor([1, 3, 7, 9]).float()
    pred = torch.tensor([2, 4, 6, 8]).float()
    loss = rank_loss(true, pred)
    assert (loss == 0).all()

    # middle pair swap, positive loss, non zero gradient for middle pair only
    true = torch.tensor([1, 3, 7, 9]).float()
    pred = torch.tensor([2, 6, 4, 8]).float().requires_grad_()
    loss = rank_loss(true, pred)
    assert (loss >= 0).all()
    loss.sum().backward()
    assert ((pred.grad != 0) == torch.tensor([False, True, True, False])).all()

    # middle pair swap, but true values under threshold, zero loss
    true = torch.tensor([1, 4.001, 4.002, 9]).float()
    pred = torch.tensor([2, 6, 4, 8]).float()
    loss = rank_loss(true, pred)
    assert (loss == 0).all()
