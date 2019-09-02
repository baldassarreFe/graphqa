import math
from typing import Tuple, Union

import torch
import numpy as np
from ignite.exceptions import NotComputableError
from ignite.metrics import Metric, MeanSquaredError


def combine_means(mean_a, mean_b, count_a, count_b):
    """Combine means.

    Args:
        mean_a:
        mean_b:
        count_a:
        count_b:

    Returns:

    """
    return (count_a * mean_a + count_b * mean_b) / (count_a + count_b)


def combine_moments(moment_a, moment_b, mean_a, mean_b, count_a, count_b):
    """Combine moments.

    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

    Args:
        moment_a:
        moment_b:
        mean_a:
        mean_b:
        count_a:
        count_b:

    Returns:

    """
    return moment_a + moment_b + (mean_a - mean_b) ** 2 * count_a * count_b / (count_a + count_b)


def combine_comoments(comoment_a, comoment_b, mean_x_a, mean_x_b, mean_y_a, mean_y_b, count_a, count_b):
    """Combine comoments.

    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online

    Args:
        comoment_a:
        comoment_b:
        mean_x_a:
        mean_x_b:
        mean_y_a:
        mean_y_b:
        count_a:
        count_b:

    Returns:

    """
    return (
            comoment_a + comoment_b +
            (mean_x_a - mean_x_b) * (mean_y_a - mean_y_b) *
            count_a * count_b / (count_a + count_b)
    )


class Sum(Metric):
    def __init__(self, output_transform=lambda x: x):
        self._total = 0
        super(Sum, self).__init__(output_transform)

    def reset(self):
        self._total = 0

    def update(self, x: Union[float, np.ndarray, torch.Tensor]):
        self._total += x

    def compute(self):
        return self._total


class Mean(Metric):
    def __init__(self, output_transform=lambda x: x):
        self._mean = 0
        self._count = 0
        super(Mean, self).__init__(output_transform)

    def reset(self):
        self._mean = 0
        self._count = 0

    def update(self, x: torch.Tensor):
        self._mean = combine_means(self._mean, x.mean().item(), self._count, len(x))
        self._count += len(x)

    def compute(self):
        if self._count < 1:
            raise NotComputableError('Mean must have at least one sample before it can be computed.')
        return self._mean


class Variance(Metric):
    def __init__(self, output_transform=lambda x: x):
        self._moment = 0
        self._mean = 0
        self._count = 0
        super(Variance, self).__init__(output_transform)

    def reset(self):
        self._moment = 0
        self._mean = 0
        self._count = 0

    def update(self, x: torch.Tensor):
        count = len(x)
        mean = x.mean().item()
        moment = (x - mean).pow(2).sum().item()

        self._moment = combine_moments(
            self._moment, moment, self._mean, mean, self._count, count)
        self._mean = combine_means(self._mean, mean, self._count, count)
        self._count += count

    def compute(self):
        if self._count < 2:
            raise NotComputableError('Variance must have at least two samples before it can be computed.')
        return self._moment / (self._count - 1)


class Covariance(Metric):
    def __init__(self, output_transform=lambda x: x):
        self._comoment = 0
        self._mean_x = 0
        self._mean_y = 0
        self._count = 0
        super(Covariance, self).__init__(output_transform)

    def reset(self):
        self._comoment = 0
        self._mean_x = 0
        self._mean_y = 0
        self._count = 0

    def update(self, x_y: Tuple[torch.Tensor, torch.Tensor]):
        x, y = x_y

        count = len(x)
        mean_x = x.mean().item()
        mean_y = y.mean().item()
        comoment = ((x - mean_x) * (y - mean_y)).sum().item()

        self._comoment = combine_comoments(
            self._comoment, comoment, self._mean_x, mean_x, self._mean_y, mean_y, self._count, count)
        self._mean_x = combine_means(self._mean_x, mean_x, self._count, count)
        self._mean_y = combine_means(self._mean_y, mean_y, self._count, count)
        self._count += count

    def compute(self):
        if self._count < 2:
            raise NotComputableError('Covariance must have at least two samples before it can be computed.')
        return self._comoment / (self._count - 1)


class PearsonR(Metric):
    def __init__(self, output_transform=lambda x: x):
        self._var_x = Covariance()
        self._var_y = Covariance()
        self._cov = Covariance()
        super(PearsonR, self).__init__(output_transform)

    def reset(self):
        self._var_x.reset()
        self._var_y.reset()
        self._cov.reset()

    def update(self, x_y: Tuple[torch.Tensor, torch.Tensor]):
        x, y = x_y
        self._var_x.update((x, x))
        self._var_y.update((y, y))
        self._cov.update((x, y))

    def compute(self):
        # If var(x) or var(y) is very small (near constant x or y vectors),
        var_x = self._var_x.compute()
        var_y = self._var_y.compute()
        denominator = math.sqrt(var_x * var_y)
        if denominator < 1e-12:
            import warnings
            from scipy.stats import PearsonRConstantInputWarning
            warnings.warn(PearsonRConstantInputWarning(
                msg=f'Denominator too small: {denominator} = sqrt({var_x} * {var_y})'))
            denominator = 1e-12
        return self._cov.compute() / max(denominator, 1e-6)


class R2(Metric):
    def __init__(self, output_transform=lambda x: x):
        self._count = 0
        self._mean_true = 0
        self._moment_true = 0
        self._sum_squared_error = 0
        super(R2, self).__init__(output_transform)

    def reset(self):
        self._count = 0
        self._mean_true = 0
        self._moment_true = 0
        self._sum_squared_error = 0

    def update(self, pred_true: Tuple[torch.Tensor, torch.Tensor]):
        pred, true = pred_true

        self._sum_squared_error += (true - pred).pow(2).sum().item()

        count = len(true)
        mean_true = true.mean().item()
        moment_true = (true - mean_true).pow(2).sum().item()
        self._moment_true = combine_moments(
            self._moment_true, moment_true, self._mean_true, mean_true, self._count, count)
        self._mean_true = combine_means(self._mean_true, mean_true, self._count, count)
        self._count += count
        pass

    def compute(self):
        return 1 - self._sum_squared_error / self._moment_true


def test():
    from tqdm import tqdm
    import torch.utils.data
    from scipy.stats import pearsonr
    from sklearn.metrics import mean_squared_error, r2_score

    device = 'cuda'
    true = torch.rand(100_000)
    pred = true + .3 * torch.randn_like(true)
    ds = torch.utils.data.TensorDataset(true, pred)

    print('Manual:')
    print('  Mean true:', true.mean().item())
    print('  Var true :', true.var().item())
    print('  Var pred :', pred.var().item())
    print('  Cov   :', np.cov(true, pred)[0, 1])
    print('  MSE   :', mean_squared_error(true, pred))
    print('  R     :', pearsonr(true, pred)[0])
    print('  R2    :', r2_score(true, pred))
    print()

    mean_true = Mean()
    var_true = Variance()
    var_pred = Variance()
    cov = Covariance()
    mse = MeanSquaredError()
    r = PearsonR()
    r2 = R2()

    for batch_size in [len(ds), 25_000, 1_000, 5, 1]:
        mean_true.reset()
        var_true.reset()
        var_pred.reset()
        cov.reset()
        mse.reset()
        r.reset()
        r2.reset()

        dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
        for true_batch, pred_batch in tqdm(dl):
            true_batch = true_batch.to(device)
            pred_batch = pred_batch.to(device)
            mean_true.update(true)
            var_true.update(true)
            var_pred.update(pred)
            cov.update((pred_batch, true_batch))
            mse.update((pred_batch, true_batch))
            r.update((pred_batch, true_batch))
            r2.update((pred_batch, true_batch))

        print(f'Batch size {batch_size}:')
        print(f'  Mean true: {mean_true.compute()} ({mean_true.compute() - true.mean().item():.0E})')
        print(f'  Var true : {var_true.compute()} ({var_true.compute() - true.var().item():.0E})')
        print(f'  Var pred : {var_pred.compute()} ({var_pred.compute() - pred.var().item():.0E})')
        print(f'  Cov   : {cov.compute()} ({cov.compute() - np.cov(true, pred)[0, 1]:.0E})')
        print(f'  MSE   : {mse.compute()} ({mse.compute() - mean_squared_error(true, pred):.0E})')
        print(f'  R     : {r.compute()} ({r.compute() - pearsonr(true, pred)[0]:.0E})')
        print(f'  R2    : {r2.compute()} ({r2.compute() - r2_score(true, pred):.0E})')
        print()


if __name__ == '__main__':
    test()
