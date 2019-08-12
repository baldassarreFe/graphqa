from collections import defaultdict
from itertools import chain
from typing import Mapping, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from torchgraphs import GraphBatch

import torch
from ignite.engine import Events
from ignite.metrics import Metric, RootMeanSquaredError, RunningAverage
from ignite.exceptions import NotComputableError

from .config import flatten_dict
from .base_metrics import combine_means, Mean, PearsonR, R2


class SafeRunningAverage(RunningAverage):
    def compute(self):
        try:
            return super(SafeRunningAverage, self).compute()
        except NotComputableError:
            return np.nan


class ProteinAverageLosses(Metric):
    def __init__(self, output_transform=lambda x: x):
        self._num_samples = 0
        self._loss_local = 0
        self._loss_global = 0
        self.losses_names = ['loss/local', 'loss/global']
        super(ProteinAverageLosses, self).__init__(output_transform)
    
    def reset(self):
        self._num_samples = 0
        self._loss_local = 0
        self._loss_global = 0

    def attach(self, engine, **kwargs):
        engine.add_event_handler(Events.EPOCH_COMPLETED, self.completed)
        if not engine.has_event_handler(self.started, Events.EPOCH_STARTED):
            engine.add_event_handler(Events.EPOCH_STARTED, self.started)
        if not engine.has_event_handler(self.iteration_completed, Events.ITERATION_COMPLETED):
            engine.add_event_handler(Events.ITERATION_COMPLETED, self.iteration_completed)
            engine.add_event_handler(Events.ITERATION_COMPLETED, self.completed)

    def update(self, output: Tuple[float, float, int]):
        loss_local, loss_global, num_samples = output
        self._loss_local = combine_means(self._loss_local, loss_local, self._num_samples, num_samples)
        self._loss_global = combine_means(self._loss_global, loss_global, self._num_samples, num_samples)
        self._num_samples += num_samples
        
    def compute(self):
        return {
            'local': self._loss_local,
            'global': self._loss_global
        }
    
    def completed(self, engine, **kwargs):
        result = self.compute()
        engine.state.metrics['loss/local'] = result['local']
        engine.state.metrics['loss/global'] = result['global']


class ProteinMetrics(Metric):
    class GlobalCorrelationPerTarget(object):
        """Fake metric, just because it was convenient to have the same methods of ignite.metrics.Metric"""

        def __init__(self):
            self.df = {'target': [], 'global_score_preds': [], 'global_score_targets': []}

        def reset(self):
            self.df = {'target': [], 'global_score_preds': [], 'global_score_targets': []}

        def update(self, protein_names: np.ndarray, global_preds: torch.Tensor, global_targets: torch.Tensor):
            self.df['target'].append(protein_names)
            self.df['global_score_preds'].append(global_preds.detach().cpu())
            self.df['global_score_targets'].append(global_targets.detach().cpu())

        def compute(self):
            return pd.DataFrame({col: np.concatenate(seq) for col, seq in self.df.items()}) \
                .groupby('target') \
                .apply(lambda df: pearsonr(df['global_score_preds'], df['global_score_targets'])[0]) \
                .mean()

    def __init__(self, output_transform=lambda x: x):
        self.local_metrics = {
            'r2': R2(),
            'rmse': RootMeanSquaredError(),
            'correlation': PearsonR(),
            'correlation_per_model': Mean(),
        }
        self.global_metrics = {
            'r2': R2(),
            'rmse': RootMeanSquaredError(),
            'correlation': PearsonR(),
            'correlation_per_target': ProteinMetrics.GlobalCorrelationPerTarget()
        }
        self.figs = {
            'local': ScoreHistogram(title='Local scores'),
            'global': ScoreHistogram(title='Global scores')
        }
        self.metric_names = [*('metric/local/' + n for n in self.local_metrics),
                             *('metric/global/' + n for n in self.global_metrics)]
        self.fig_names = ['fig/' + n for n in self.figs]
        super(ProteinMetrics, self).__init__(output_transform)
        
    def reset(self):
        for metric in chain(self.local_metrics.values(), self.global_metrics.values(), self.figs.values()):
            metric.reset()

    def attach(self, engine, **kwargs):
        super(ProteinMetrics, self).attach(engine, None)

    def update(self, output: Tuple[List, List, GraphBatch, GraphBatch]):
        protein_names, model_names, preds, targets = output

        # All metrics: skip native structures
        non_native = torch.tensor(np.char.not_equal(model_names, 'native'), device=preds.node_features.device)

        # Local metrics: ignore residues that don't have a ground-truth score 
        valid_scores = torch.isfinite(targets.node_features[:, 0])
        ix = torch.repeat_interleave(non_native, repeats=preds.num_nodes_by_graph) & valid_scores
        node_preds = preds.node_features[ix, 0]
        node_targets = targets.node_features[ix, 0]
        self.local_metrics['r2'].update((node_preds, node_targets))
        self.local_metrics['rmse'].update((node_preds, node_targets))
        self.local_metrics['correlation'].update((node_preds, node_targets))
        self.figs['local'].update((node_preds, node_targets))

        # Local correlation per model is a bit more complex, pandas is the easiest way to get a groupby
        # This dataframe is already constructed without native structures and NaN targets
        correlations_per_model = pd.DataFrame({
            # Used to uniquely identify a (protein, model) pair without using their str names
            'target_model': preds.node_index_by_graph[ix].cpu(),
            'node_preds': node_preds.cpu(),
            'node_targets': node_targets.cpu()
        }).groupby('target_model').apply(lambda df: pearsonr(df['node_preds'], df['node_targets'])[0])
        self.local_metrics['correlation_per_model'].update(torch.from_numpy(correlations_per_model.values))

        # Global metrics 
        global_preds = preds.global_features[non_native].squeeze()
        global_targets = targets.global_features[non_native].squeeze()
        self.global_metrics['r2'].update((global_preds, global_targets))
        self.global_metrics['rmse'].update((global_preds, global_targets))
        self.global_metrics['correlation'].update((global_preds, global_targets))
        self.global_metrics['correlation_per_target'].update(
            np.array(protein_names)[non_native.cpu().numpy().astype(np.bool)], global_preds, global_targets)
        self.figs['global'].update((global_preds, global_targets))

    def compute(self):
        metrics = {
            'local': {name: metric.compute() for name, metric in self.local_metrics.items()},
            'global': {name: metric.compute() for name, metric in self.global_metrics.items()}
        }
        figs = {
            'local': self.figs['local'].compute(metrics['local']['correlation'], metrics['local']['r2']),
            'global': self.figs['global'].compute(metrics['global']['correlation'], metrics['global']['r2'])
        }
        return {'metric': metrics, 'fig': figs}

    def completed(self, engine, prefix):
        result = self.compute()
        for keys, value in flatten_dict(result):
            engine.state.metrics['/'.join(keys)] = value


class ScoreHistogram(Metric):
    def __init__(self, bins=100, title=None):
        self.hist = np.empty((bins, bins), dtype=np.float)
        self.bins = np.linspace(0, 1, bins + 1)
        self.title = title
        super(ScoreHistogram, self).__init__()

    def reset(self):
        self.hist.fill(0)

    def update(self, pred_true: Tuple[torch.Tensor, torch.Tensor]):
        pred, true = pred_true
        hist, _, _ = np.histogram2d(true.cpu().numpy(), pred.cpu().numpy(), bins=self.bins)
        self.hist += hist

    def compute(self, pearson_r=None, r_squared=None):
        fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=100)
        ax.pcolormesh(self.bins, self.bins, self.hist.T)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('True')
        ax.set_ylabel('Predicted')
        ax.plot([0, 1], [0, 1], color='red', linestyle='--')

        title = self.title.capitalize() if self.title is not None else ''
        if pearson_r is not None:
            title += f' R: {pearson_r:.3f}'
        if r_squared is not None:
            title += f' R2: {r_squared:.3f}'
        ax.set_title(title)

        return fig

    def completed(self, engine, name):
        pearson_r = engine.state.metrics.get('metric/' + name + '/correlation')
        r_squared = engine.state.metrics.get('metric/' + name + '/r2')
        result = self.compute(pearson_r, r_squared)
        engine.state.metrics['fig/' + name] = result


class LocalCorrelationPerModelSlow(Metric):
    def __init__(self, output_transform=lambda x: x):
        self._targets: Mapping[str, Mapping[str, PearsonR]] = defaultdict(lambda: defaultdict(lambda: PearsonR()))
        super(LocalCorrelationPerModelSlow, self).__init__(output_transform)

    def reset(self):
        self._targets = defaultdict(lambda: defaultdict(lambda: PearsonR()))

    def update(self, output):
        for target, model, preds, targets in zip(*output):
            if model != 'native':
                node_mask = torch.isfinite(targets.node_features[:, 0])
                self._targets[target][model].update((
                    preds.node_features[node_mask, 0], targets.node_features[node_mask, 0]))

    def compute(self):
        return np.mean([model.compute() for target in self._targets.values() for model in target.values()])


class GlobalCorrelationPerTargetSlow(Metric):
    def __init__(self, output_transform=lambda x: x):
        self._targets: Mapping[str, PearsonR] = defaultdict(lambda: PearsonR())
        super(GlobalCorrelationPerTargetSlow, self).__init__(output_transform)

    def reset(self):
        self._targets = defaultdict(lambda: PearsonR())

    def update(self, output):
        for target, model, preds, targets in zip(*output):
            if model != 'native':
                self._targets[target].update((preds.global_features, targets.global_features))

    def compute(self):
        return sum(target.compute() for target in self._targets.values()) / len(self._targets)


def test():
    import os
    from pathlib import Path

    import pyaml
    import torch.utils.data
    import torchgraphs as tg
    from tqdm import tqdm
    from sklearn.metrics import mean_squared_error, r2_score

    from proteins.dataset import ProteinFolder

    metrics = ProteinMetrics()

    local_correlation_per_model = LocalCorrelationPerModelSlow()
    global_correlation_per_target = GlobalCorrelationPerTargetSlow()

    dataloader_val = torch.utils.data.DataLoader(
        ProteinFolder(Path(os.environ.get('DATA_FOLDER', './data')) / 'validation', cutoff=8),
        shuffle=True, batch_size=100, collate_fn=tg.GraphBatch.collate
    )

    device = 'cpu'
    nodes_df = defaultdict(list)
    graphs_df = defaultdict(list)

    for target_name, model_name, graphs, targets in tqdm(dataloader_val):
        graphs = graphs.to(device)
        preds = graphs.evolve(
            node_features=(targets.node_features[:, 0] + .3 * torch.randn_like(targets.node_features[:, 0]))[:, None],
            global_features=targets.global_features + .2 * torch.randn_like(targets.global_features)
        )

        metrics.update((target_name, model_name, preds, targets))
        local_correlation_per_model.update((target_name, model_name, preds, targets))
        global_correlation_per_target.update((target_name, model_name, preds, targets))

        nodes_df['ProteinName'].append(np.repeat(np.array(target_name), graphs.num_nodes_by_graph.cpu()))
        nodes_df['ModelName'].append(np.repeat(np.array(model_name), graphs.num_nodes_by_graph.cpu()))
        nodes_df['NodeId'].append(np.concatenate([np.arange(n) for n in graphs.num_nodes_by_graph.cpu()]))
        nodes_df['LocalScoreTrue'].append(targets.node_features[:, 0].cpu())
        nodes_df['LocalScorePredicted'].append(preds.node_features.squeeze().cpu())

        graphs_df['ProteinName'].append(np.array(target_name))
        graphs_df['ModelName'].append(np.array(model_name))
        graphs_df['GlobalScoreTrue'].append(targets.global_features.squeeze(dim=1).cpu())
        graphs_df['GlobalScorePredicted'].append(preds.global_features.squeeze(dim=1).cpu())

    metrics, figs = metrics.compute().values()
    metrics['local']['correlation_per_model_slow'] = local_correlation_per_model.compute()
    metrics['global']['pearson_R_per_target_slow'] = global_correlation_per_target.compute()

    print(pyaml.dump({'ignite': metrics}, safe=True))
    figs['local'].show()
    figs['global'].show()

    nodes_df = pd.DataFrame({k: np.concatenate(v) for k, v in nodes_df.items()}).query('ModelName != "native"')
    metrics = defaultdict(dict)

    def nanrmse(true, predicted):
        mask = np.isfinite(true)
        return np.sqrt(mean_squared_error(true[mask], predicted[mask]))

    def nanpearsonr(true, predicted):
        mask = np.isfinite(true)
        return pearsonr(true[mask], predicted[mask])[0]

    def nanr2(true, predicted):
        mask = np.isfinite(true)
        return r2_score(true[mask], predicted[mask])

    metrics['local']['rmse'] = nanrmse(nodes_df['LocalScoreTrue'], nodes_df['LocalScorePredicted'])
    metrics['local']['r2'] = nanr2(nodes_df['LocalScoreTrue'], nodes_df['LocalScorePredicted'])
    metrics['local']['correlation'] = nanpearsonr(nodes_df['LocalScoreTrue'], nodes_df['LocalScorePredicted'])
    metrics['local']['correlation_per_model'] = nodes_df.groupby(['ProteinName', 'ModelName']) \
        .apply(lambda df: nanpearsonr(df['LocalScoreTrue'], df['LocalScorePredicted'])) \
        .mean()

    graphs_df = pd.DataFrame({k: np.concatenate(v) for k, v in graphs_df.items()}).query('ModelName != "native"')

    metrics['globals']['rmse'] = nanrmse(graphs_df['GlobalScoreTrue'], graphs_df['GlobalScorePredicted'])
    metrics['globals']['r2'] = nanr2(graphs_df['GlobalScoreTrue'], graphs_df['GlobalScorePredicted'])
    metrics['globals']['correlation'] = pearsonr(graphs_df['GlobalScoreTrue'], graphs_df['GlobalScorePredicted'])[0]
    metrics['globals']['correlation_per_target'] = graphs_df.groupby('ProteinName') \
        .apply(lambda df: pearsonr(df['GlobalScoreTrue'], df['GlobalScorePredicted'])[0]) \
        .mean()

    print(pyaml.dump({'pandas': metrics}, safe=True))

    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=100)
    ax.hist2d(nodes_df['LocalScoreTrue'], nodes_df['LocalScorePredicted'], bins=np.linspace(0, 1, 100+1))
    ax.set_title(f'Local Scores (R: {metrics["local"]["correlation"]:.3f} R2: {metrics["local"]["r2"]:.3f})')
    ax.set_xlabel('True')
    ax.set_ylabel('Predicted')
    ax.plot([0, 1], [0, 1], color='red', linestyle='--')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.show()

    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=100)
    ax.hist2d(graphs_df['GlobalScoreTrue'], graphs_df['GlobalScorePredicted'], bins=np.linspace(0, 1, 100+1))
    ax.set_title(f'Global Scores (R: {metrics["globals"]["correlation"]:.3f} R2: {metrics["globals"]["r2"]:.3f})')
    ax.set_xlabel('True')
    ax.set_ylabel('Predicted')
    ax.plot([0, 1], [0, 1], color='red', linestyle='--')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.show()


if __name__ == '__main__':
    test()
