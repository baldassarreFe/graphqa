import torch


@torch.jit.script
def nan_mse(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Mean squared error loss that ignores NaN values in `targets` tensor.
    Args:
        inputs:
        targets:

    Returns:

    """
    mask = torch.isfinite(targets)
    zero = targets.new_zeros(())
    num = torch.where(mask, targets - inputs, zero).pow(2).sum(dim=0)
    den = mask.sum(dim=0)
    return num / den
