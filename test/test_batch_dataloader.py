from math import ceil

import torch
import torch.utils.data

from proteins.dataset import TargetBatchSampler


def test_batch_dataloader():
    lengths = [
        4,   # smaller than batch size
        8,   # one batch + 3 spare
        10,  # two full batches
        12   # two batches + 2 spare
    ]
    batch_size = 5

    ds = torch.utils.data.ConcatDataset([
        torch.utils.data.TensorDataset(torch.tensor([(i, j) for j in range(l)]))
        for i, l in enumerate(lengths)
    ])

    # Drop last = False
    dl = torch.utils.data.DataLoader(
        ds,
        batch_sampler=TargetBatchSampler(ds, batch_size=batch_size, shuffle=True, drop_last=False)
    )
    assert len(dl) == sum(int(ceil(l / batch_size)) for l in lengths)

    num_batches = 0
    num_samples = 0
    for x, *_ in iter(dl):
        num_batches += 1
        num_samples += len(x)
        assert torch.all(x[:, 0] == x[0, 0]).item()

    assert num_samples == len(ds)
    assert num_batches == sum(int(ceil(l / batch_size)) for l in lengths)

    # Drop last = True
    dl = torch.utils.data.DataLoader(
        ds,
        batch_sampler=TargetBatchSampler(ds, batch_size=batch_size, shuffle=True, drop_last=True)
    )
    assert len(dl) == sum(l // batch_size for l in lengths)

    num_batches = 0
    num_samples = 0
    for x, *_ in iter(dl):
        num_batches += 1
        num_samples += len(x)
        assert torch.all(x[:, 0] == x[0, 0]).item()

    assert num_samples == sum(batch_size * (l // batch_size) for l in lengths)
    assert num_batches == sum(l // batch_size for l in lengths)
