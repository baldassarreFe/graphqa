from typing import Tuple, Union
from collections import defaultdict

import scipy.stats
import sklearn.metrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import ignite
from ignite.engine import Events
from ignite.metrics import Metric, RootMeanSquaredError

from . import features
from .base_metrics import combine_means, Mean, PearsonR


def rmse(preds, true):
    return np.sqrt(sklearn.metrics.mean_squared_error(preds, true))


def r2_score(preds, true):
    return sklearn.metrics.r2_score(true, preds)


def pearson(preds, true):
    return scipy.stats.pearsonr(preds, true)[0]


def spearmanr(preds, true):
    return scipy.stats.spearmanr(preds, true)[0]


def kendalltau(preds, true):
    return scipy.stats.kendalltau(preds, true)[0]


def first_rank_loss(preds, true):
    return true.max() - true[preds.idxmax()]


def customize_state(engine):
    """Add `figures`, `losses`, `misc` to the ignite.engine.State instance that is created after starting the Engine"""
    setattr(engine.state, 'figures', {})
    setattr(engine.state, 'losses', {})
    setattr(engine.state, 'misc', {})


class GpuMaxMemoryAllocated(Metric):
    """Max GPU memory allocated in MB"""

    def update(self, output):
        pass

    def reset(self):
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_max_memory_allocated(i)

    def compute(self):
        return max(torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())) // 2**20

    def attach(self, engine, name):
        engine.add_event_handler(Events.EPOCH_COMPLETED, self.completed, name)
        if not engine.has_event_handler(self.started, Events.EPOCH_STARTED):
            engine.add_event_handler(Events.EPOCH_STARTED, self.started)

    def completed(self, engine, name):
        result = self.compute()
        engine.state.misc[name] = result


class LocalMetrics(ignite.metrics.Metric):
    METRICS = (
        'rmse',
        'pearson',
        'per_model_pearson',
    )
    FIGURES = (
        'hist',
    )

    def __init__(self, column, title=None, metrics=None, figures=None, output_transform=lambda x: x):
        self.column = column
        self.title = title if title is not None else ''
        self.metrics = set(metrics if metrics is not None else LocalMetrics.METRICS)
        self.figures = set(figures if figures is not None else LocalMetrics.FIGURES)
        self._rmse = RootMeanSquaredError()
        self._pearson = PearsonR()
        self._per_model_pearson = Mean()
        self._hist = ScoreHistogram(title=title)
        super(LocalMetrics, self).__init__(output_transform=output_transform)

    def reset(self):
        self._rmse.reset()
        self._pearson.reset()
        self._per_model_pearson.reset()
        self._hist.reset()

    def update(self, output):
        protein_names, model_names, preds, targets = output

        # Skip native structures and ignore residues that don't have a ground-truth score
        non_native = np.repeat(np.char.not_equal(model_names, 'native'), repeats=preds.num_nodes_by_graph.cpu().numpy())
        has_score = torch.isfinite(targets.node_features[:, self.column]).cpu().numpy()
        valid_scores = np.logical_and(non_native, has_score)

        # Used to uniquely identify a (protein, model) pair without using their str names
        target_model_id = preds.node_index_by_graph[valid_scores].cpu().numpy()
        node_preds = preds.node_features[valid_scores, self.column].detach().cpu().numpy()
        node_targets = targets.node_features[valid_scores, self.column].detach().cpu().numpy()

        # Streaming metrics on local scores (they expect torch tensors, not numpy arrays)
        self._rmse.update((torch.from_numpy(node_preds), torch.from_numpy(node_targets)))
        self._pearson.update((torch.from_numpy(node_preds), torch.from_numpy(node_targets)))

        # Per model metrics: pandas is the easiest way to get a groupby.
        grouped = pd.DataFrame({
            'target_model': target_model_id,
            'preds': node_preds,
            'true': node_targets
        }).groupby('target_model')

        per_model_pearsons = grouped.apply(lambda df: pearson(df['preds'], df['true']))
        self._per_model_pearson.update(torch.from_numpy(per_model_pearsons.values))

        self._hist.update(node_preds, node_targets)

    def compute(self):
        metrics = {}
        figures = {}

        if 'rmse' in self.metrics:
            metrics['rmse'] = self._rmse.compute()
        if 'pearson' in self.metrics:
            metrics['pearson'] = self._pearson.compute()
        if 'per_model_pearson' in self.metrics:
            metrics['per_model_pearson'] = self._per_model_pearson.compute()

        if 'hist' in self.figures:
            extra_title = ''
            if 'pearson' in self.metrics:
                extra_title += f'$R$ {metrics["pearson"]:.3f} '
            if 'per_model_pearson' in self.metrics:
                extra_title += f'$R_{{model}}$ {metrics["per_model_pearson"]:.3f} '
            figures['hist'] = self._hist.compute(extra_title.strip())

        return {'metrics': metrics, 'figures': figures}

    def completed(self, engine, prefix):
        result = self.compute()
        for name, metric in result['metrics'].items():
            engine.state.metrics[prefix + '/' + name] = metric
        for name, fig in result['figures'].items():
            engine.state.figures[prefix + '/' + name] = fig


class GlobalMetrics(ignite.metrics.Metric):
    METRICS = (
        'rmse',
        'pearson',
        'spearman',
        'kendall',
        'per_target_pearson',
        'per_target_spearman',
        'per_target_kendall',
        'first_rank_loss',
    )
    FIGURES = (
        'hist',
        'funnel',
    )

    def __init__(self, column, title=None, metrics=None, figures=None, output_transform=lambda x: x):
        self.column = column
        self.title = title if title is not None else ''
        self.metrics = set(metrics if metrics is not None else GlobalMetrics.METRICS)
        self.figures = set(figures if figures is not None else GlobalMetrics.FIGURES)
        self._lists = {'target': [], 'preds': [], 'true': []}
        self._hist = ScoreHistogram(title=title)
        super(GlobalMetrics, self).__init__(output_transform=output_transform)

    def reset(self):
        for l in self._lists.values():
            l.clear()
        self._hist.reset()

    def update(self, output):
        protein_names, model_names, results, targets = output

        # Skip native structures
        non_native = np.char.not_equal(model_names, 'native')

        protein_names = np.array(protein_names)[non_native]
        preds = results.global_features[non_native, self.column].detach().cpu().numpy()
        true = targets.global_features[non_native, self.column].detach().cpu().numpy()

        self._lists['target'].append(protein_names)
        self._lists['preds'].append(preds)
        self._lists['true'].append(true)

    @staticmethod
    def _funnel(grouped, ncols):
        nrows = int(np.ceil(grouped.ngroups / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(2 * ncols, 2 * nrows), dpi=100)

        for ax, (target_name, group) in zip(axes.ravel(), grouped):
            best_true = group['true'].idxmax()
            best_preds = group['preds'].idxmax()
            ax.plot([0, 1], [0, 1], color='k', linestyle='--', linewidth=.5, zorder=1)
            ax.scatter(data=group, x='true', y='preds', marker='.', zorder=2)
            ax.scatter(group['true'][best_preds], group['preds'][best_preds], color='r', marker='.', zorder=3)
            ax.axvline(group['true'][best_preds], color='r', linewidth=.7, zorder=3)
            ax.scatter(group['true'][best_true], group['preds'][best_true], color='g', marker='.', zorder=4)
            ax.axvline(group['true'][best_true], color='g', linewidth=.7, zorder=4)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(target_name)

        axes[-1, 0].set_xticks([0, .5, 1.])
        axes[-1, 0].set_yticks([0, .5, 1.])
        axes[-1, 0].set_xlabel('True')
        axes[-1, 0].set_ylabel('Predicted')

        # Hide unused axes
        for ax in axes.ravel()[grouped.ngroups:]:
            ax.set_visible(False)

        fig.tight_layout()
        return fig

    def compute(self):
        metrics = {}
        figures = {}
        df = pd.DataFrame({col: np.concatenate(seq) for col, seq in self._lists.items()})
        grouped = df.groupby('target', sort=True)

        if 'rmse' in self.metrics:
            metrics['rmse'] = rmse(df['preds'], df['true'])
        if 'pearson' in self.metrics:
            metrics['pearson'] = pearson(df['preds'], df['true'])
        if 'spearman' in self.metrics:
            metrics['spearman'] = spearmanr(df['preds'], df['true'])
        if 'kendall' in self.metrics:
            metrics['kendall'] = kendalltau(df['preds'], df['true'])

        if 'per_target_pearson' in self.metrics:
            metrics['per_target_pearson'] = grouped.apply(lambda g: pearson(g['preds'], g['true'])).mean()
        if 'per_target_spearman' in self.metrics:
            metrics['per_target_spearman'] = grouped.apply(lambda g: spearmanr(g['preds'], g['true'])).mean()
        if 'per_target_kendall' in self.metrics:
            metrics['per_target_kendall'] = grouped.apply(lambda g: kendalltau(g['preds'], g['true'])).mean()
        if 'first_rank_loss' in self.metrics:
            metrics['first_rank_loss'] = grouped.apply(lambda g: first_rank_loss(g['preds'], g['true'])).mean()

        if 'hist' in self.figures:
            extra_title = ''
            if 'pearson' in self.metrics:
                extra_title += f'$R {metrics["pearson"]:.3f}$ '
            if 'per_target_pearson' in self.metrics:
                extra_title += f'$R_{{target}} {metrics["per_target_pearson"]:.3f}$ '
            figures['hist'] = self._hist.update(df['preds'], df['true']).compute(extra_title.strip())
        if 'funnel' in self.figures:
            figures['funnel'] = self._funnel(grouped, ncols=8)

        return {'metrics': metrics, 'figures': figures}

    def completed(self, engine, prefix):
        result = self.compute()
        for name, metric in result['metrics'].items():
            engine.state.metrics[prefix + '/' + name] = metric
        for name, fig in result['figures'].items():
            engine.state.figures[prefix + '/' + name] = fig


class ScoreHistogram(object):
    def __init__(self, bins=100, title=None):
        self.hist = np.empty((bins, bins), dtype=np.float)
        self.bins = np.linspace(0, 1, bins + 1)
        self.title = title

    def reset(self):
        self.hist.fill(0)

    def update(self, preds: Union[torch.Tensor, np.ndarray], true: Union[torch.Tensor, np.ndarray]):
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
        if isinstance(true, torch.Tensor):
            true = true.detach().cpu().numpy()

        hist, _, _ = np.histogram2d(true, preds, bins=self.bins)
        self.hist += hist

        return self

    def compute(self, extra_title=None):
        fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=100)
        ax.pcolormesh(self.bins, self.bins, self.hist.T)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('True')
        ax.set_ylabel('Predicted')
        ax.plot([0, 1], [0, 1], color='red', linestyle='--')
        ax.set_title(' '.join(filter(lambda string: string is not None and len(string) > 0, (self.title, extra_title))))
        return fig


# noinspection PyAttributeOutsideInit
class ProteinAverageLosses(Metric):
    losses_names = ['local_lddt', 'global_lddt', 'global_gdtts']

    def reset(self):
        self._num_samples = 0
        self._loss_local_lddt = 0
        self._loss_global_lddt = 0
        self._loss_global_gdtts = 0

    def attach(self, engine, **ignored_kwargs):
        engine.add_event_handler(Events.EPOCH_COMPLETED, self.completed)
        if not engine.has_event_handler(self.started, Events.EPOCH_STARTED):
            engine.add_event_handler(Events.EPOCH_STARTED, self.started)
        if not engine.has_event_handler(self.iteration_completed, Events.ITERATION_COMPLETED):
            engine.add_event_handler(Events.ITERATION_COMPLETED, self.iteration_completed)

    def update(self, output: Tuple[float, float, float, int]):
        loss_local_lddt, loss_global_lddt, loss_global_gdtts, samples = output
        self._loss_local_lddt = combine_means(self._loss_local_lddt, loss_local_lddt, self._num_samples, samples)
        self._loss_global_lddt = combine_means(self._loss_global_lddt, loss_global_lddt, self._num_samples, samples)
        self._loss_global_gdtts = combine_means(self._loss_global_gdtts, loss_global_gdtts, self._num_samples, samples)
        self._num_samples += samples

    def compute(self):
        return {
            'local_lddt': self._loss_local_lddt,
            'global_lddt': self._loss_global_lddt,
            'global_gdtts': self._loss_global_gdtts,
        }

    def completed(self, engine, **ignored_kwargs):
        result = self.compute()
        engine.state.losses.update(result)


def test():
    """Go through the validation dataset, output fake predictions using random noise,
    compute all metrics of interest using ignite-based classes and good old pandas,
    make sure that the results and the figures match."""
    import os
    from pathlib import Path

    import pyaml
    import torch.utils.data
    import torchgraphs as tg
    from tqdm import tqdm

    from .config import flatten_dict
    from .dataset import ProteinQualityDataset, RemoveEdges

    samples_df = pd.read_csv(Path(os.environ.get('DATA_FOLDER', './data')) / 'samples.csv')
    samples_df['path'] = [Path(os.environ.get('DATA_FOLDER', './data')) / p for p in samples_df['path']]
    dataloader = torch.utils.data.DataLoader(
        ProteinQualityDataset(samples_df, transforms=[RemoveEdges(cutoff=6)]),
        shuffle=True, batch_size=100, collate_fn=tg.GraphBatch.collate
    )

    local_lddt_metrics = LocalMetrics(features.Output.Node.LOCAL_LDDT, title='Local LDDT')
    global_lddt_metrics = GlobalMetrics(features.Output.Global.GLOBAL_LDDT, title='Global LDDT')
    global_gdtts_metrics = GlobalMetrics(features.Output.Global.GLOBAL_GDTTS, title='Global GDT_TS')

    device = 'cuda'
    nodes_df = defaultdict(list)
    graphs_df = defaultdict(list)

    for target_name, model_name, graphs, targets in tqdm(dataloader):
        graphs = graphs.to(device)
        targets = targets.to(device)

        node_features = targets.node_features[:, :features.Output.Node.LENGTH]
        node_features = node_features + .1 * torch.randn_like(node_features)
        node_features = (np.round(2 * node_features.cpu(), decimals=1) / 2).clamp(min=0, max=1)
        global_features = targets.global_features[:, :features.Output.Global.LENGTH]
        global_features = global_features + .1 * torch.randn_like(global_features)
        global_features = (np.round(2 * global_features.cpu(), decimals=1) / 2).clamp(min=0, max=1)
        preds = graphs.evolve(node_features=node_features.to(device), global_features=global_features.to(device))

        local_lddt_metrics.update((target_name, model_name, preds, targets))
        global_lddt_metrics.update((target_name, model_name, preds, targets))
        global_gdtts_metrics.update((target_name, model_name, preds, targets))

        nodes_df['ProteinName'].append(np.repeat(np.array(target_name), graphs.num_nodes_by_graph.cpu()))
        nodes_df['ModelName'].append(np.repeat(np.array(model_name), graphs.num_nodes_by_graph.cpu()))
        nodes_df['NodeId'].append(np.concatenate([np.arange(n) for n in graphs.num_nodes_by_graph.cpu()]))
        nodes_df['LocalLddtTrue'].append(targets.node_features[:, features.Output.Node.LOCAL_LDDT].cpu())
        nodes_df['LocalLddtPred'].append(preds.node_features[:, features.Output.Node.LOCAL_LDDT].cpu())

        graphs_df['ProteinName'].append(np.array(target_name))
        graphs_df['ModelName'].append(np.array(model_name))
        graphs_df['GlobalLddtTrue'].append(targets.global_features[:, features.Output.Global.GLOBAL_LDDT].cpu())
        graphs_df['GlobalLddtPred'].append(preds.global_features[:, features.Output.Global.GLOBAL_LDDT].cpu())
        graphs_df['GlobalGdttsTrue'].append(targets.global_features[:, features.Output.Global.GLOBAL_GDTTS].cpu())
        graphs_df['GlobalGdttsPred'].append(preds.global_features[:, features.Output.Global.GLOBAL_GDTTS].cpu())

    local_lddt_metrics = local_lddt_metrics.compute()
    global_lddt_metrics = global_lddt_metrics.compute()
    global_gdtts_metrics = global_gdtts_metrics.compute()
    metrics = {
        'local_lddt': local_lddt_metrics['metrics'],
        'global_lddt': global_lddt_metrics['metrics'],
        'global_gdtts': global_gdtts_metrics['metrics'],
    }
    figures = {
        'local_lddt': local_lddt_metrics['figures'],
        'global_lddt': global_lddt_metrics['figures'],
        'global_gdtts': global_gdtts_metrics['figures'],
    }

    print(pyaml.dump({'ignite': metrics}, safe=True, sort_dicts=False))
    for keys, fig in flatten_dict(figures):
        fig.show()

    figures['global_lddt']['funnel'].savefig('/home/federico/Desktop/fig.png')
    figures['global_lddt']['funnel'].savefig('/home/federico/Desktop/fig.pdf')

    metrics = defaultdict(dict)
    nodes_df = pd.DataFrame({k: np.concatenate(v) for k, v in nodes_df.items()}) \
        .query('ModelName != "native"')\
        .dropna(subset=['LocalLddtTrue'])
    graphs_df = pd.DataFrame({k: np.concatenate(v) for k, v in graphs_df.items()}) \
        .query('ModelName != "native"')

    metrics['local_lddt']['rmse'] = rmse(nodes_df['LocalLddtPred'], nodes_df['LocalLddtTrue'])
    metrics['local_lddt']['pearson'] = pearson(nodes_df['LocalLddtPred'], nodes_df['LocalLddtTrue'])
    metrics['local_lddt']['per_model_pearson'] = nodes_df.groupby(['ProteinName', 'ModelName']) \
        .apply(lambda df: pearson(df['LocalLddtPred'], df['LocalLddtTrue'])) \
        .mean()

    metrics['global_lddt']['rmse'] = rmse(graphs_df['GlobalLddtPred'], graphs_df['GlobalLddtTrue'])
    metrics['global_lddt']['pearson'] = pearson(graphs_df['GlobalLddtPred'], graphs_df['GlobalLddtTrue'])
    metrics['global_lddt']['spearman'] = spearmanr(graphs_df['GlobalLddtPred'], graphs_df['GlobalLddtTrue'])
    metrics['global_lddt']['kendall'] = kendalltau(graphs_df['GlobalLddtPred'], graphs_df['GlobalLddtTrue'])
    metrics['global_lddt']['per_target_pearson'] = graphs_df.groupby('ProteinName') \
        .apply(lambda df: pearson(df['GlobalLddtPred'], df['GlobalLddtTrue'])) \
        .mean()
    metrics['global_lddt']['per_target_spearman'] = graphs_df.groupby('ProteinName') \
        .apply(lambda df: spearmanr(df['GlobalLddtPred'], df['GlobalLddtTrue'])) \
        .mean()
    metrics['global_lddt']['per_target_kendall'] = graphs_df.groupby('ProteinName') \
        .apply(lambda df: kendalltau(df['GlobalLddtPred'], df['GlobalLddtTrue'])) \
        .mean()
    metrics['global_lddt']['first_rank_loss'] = graphs_df.groupby('ProteinName') \
        .apply(lambda df: first_rank_loss(df['GlobalLddtPred'], df['GlobalLddtTrue'])) \
        .mean()

    metrics['global_gdtts']['rmse'] = rmse(graphs_df['GlobalGdttsPred'], graphs_df['GlobalGdttsTrue'])
    metrics['global_gdtts']['r2'] = r2_score(graphs_df['GlobalGdttsPred'], graphs_df['GlobalGdttsTrue'])
    metrics['global_gdtts']['pearson'] = pearson(graphs_df['GlobalGdttsPred'], graphs_df['GlobalGdttsTrue'])
    metrics['global_gdtts']['spearman'] = spearmanr(graphs_df['GlobalGdttsPred'], graphs_df['GlobalGdttsTrue'])
    metrics['global_gdtts']['kendall'] = kendalltau(graphs_df['GlobalGdttsPred'], graphs_df['GlobalGdttsTrue'])
    metrics['global_gdtts']['per_target_pearson'] = graphs_df.groupby('ProteinName') \
        .apply(lambda df: pearson(df['GlobalGdttsPred'], df['GlobalGdttsTrue'])) \
        .mean()
    metrics['global_gdtts']['per_target_spearman'] = graphs_df.groupby('ProteinName') \
        .apply(lambda df: spearmanr(df['GlobalGdttsPred'], df['GlobalGdttsTrue'])) \
        .mean()
    metrics['global_gdtts']['per_target_kendall'] = graphs_df.groupby('ProteinName') \
        .apply(lambda df: kendalltau(df['GlobalGdttsPred'], df['GlobalGdttsTrue'])) \
        .mean()
    metrics['global_gdtts']['first_rank_loss'] = graphs_df.groupby('ProteinName') \
        .apply(lambda df: first_rank_loss(df['GlobalGdttsPred'], df['GlobalGdttsTrue'])) \
        .mean()

    print(pyaml.dump({'pandas': metrics}, safe=True, sort_dicts=False))

    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=100)
    ax.hist2d(nodes_df['LocalLddtTrue'], nodes_df['LocalLddtPred'], bins=np.linspace(0, 1, 100+1))
    ax.set_title(f'Local LDDT $R$: {metrics["local_lddt"]["pearson"]:.3f} $R_{{model}}$: {metrics["local_lddt"]["per_model_pearson"]:.3f}')
    ax.set_xlabel('True')
    ax.set_ylabel('Predicted')
    ax.plot([0, 1], [0, 1], color='red', linestyle='--')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.show()

    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=100)
    ax.hist2d(graphs_df['GlobalLddtTrue'], graphs_df['GlobalLddtPred'], bins=np.linspace(0, 1, 100+1))
    ax.set_title(f'Global LDDT R: {metrics["global_lddt"]["pearson"]:.3f} $R_{{target}}$: {metrics["global_lddt"]["per_target_pearson"]:.3f}')
    ax.set_xlabel('True')
    ax.set_ylabel('Predicted')
    ax.plot([0, 1], [0, 1], color='red', linestyle='--')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.show()

    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=100)
    ax.hist2d(graphs_df['GlobalGdttsTrue'], graphs_df['GlobalGdttsPred'], bins=np.linspace(0, 1, 100+1))
    ax.set_title(f'Global GDT_TS R: {metrics["global_gdtts"]["pearson"]:.3f} $R_{{target}}$: {metrics["global_gdtts"]["per_target_pearson"]:.3f}')
    ax.set_xlabel('True')
    ax.set_ylabel('Predicted')
    ax.plot([0, 1], [0, 1], color='red', linestyle='--')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.show()


if __name__ == '__main__':
    test()
