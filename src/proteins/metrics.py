from itertools import chain
from collections import defaultdict
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

from . import features
from .config import flatten_dict
from .base_metrics import combine_means, Mean, PearsonR, R2


class SafeRunningAverage(RunningAverage):
    def compute(self):
        try:
            return super(SafeRunningAverage, self).compute()
        except NotComputableError:
            return np.nan


# noinspection PyAttributeOutsideInit
class ProteinAverageLosses(Metric):
    losses_names = ['loss/local_lddt', 'loss/global_lddt', 'loss/global_gdtts']

    def reset(self):
        self._num_samples = 0
        self._loss_local_lddt = 0
        self._loss_global_lddt = 0
        self._loss_global_gdtts = 0

    def attach(self, engine, **kwargs):
        engine.add_event_handler(Events.EPOCH_COMPLETED, self.completed)
        if not engine.has_event_handler(self.started, Events.EPOCH_STARTED):
            engine.add_event_handler(Events.EPOCH_STARTED, self.started)
        if not engine.has_event_handler(self.iteration_completed, Events.ITERATION_COMPLETED):
            engine.add_event_handler(Events.ITERATION_COMPLETED, self.iteration_completed)
            engine.add_event_handler(Events.ITERATION_COMPLETED, self.completed)

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
    
    def completed(self, engine, **kwargs):
        result = self.compute()
        engine.state.metrics['loss/local_lddt'] = result['local_lddt']
        engine.state.metrics['loss/global_lddt'] = result['global_lddt']
        engine.state.metrics['loss/global_gdtts'] = result['global_gdtts']


class ProteinMetrics(Metric):
    # noinspection PyAttributeOutsideInit
    class PerTargetMetrics(object):
        """Fake metric, just because it was convenient to have the same methods of ignite.metrics.Metric"""
        metric_names = ['correlation_per_target', 'first_rank_loss']

        def reset(self):
            self.df = {'target': [], 'preds': [], 'targets': []}

        def update(self, protein_names: np.ndarray, preds: torch.Tensor, targets: torch.Tensor):
            self.df['target'].append(protein_names)
            self.df['preds'].append(preds.detach().cpu())
            self.df['targets'].append(targets.detach().cpu())

        @staticmethod
        def _first_rank_loss(predicted, true):
            return true.max() - true[predicted.idxmax()]

        def compute(self):
            grouped = pd.DataFrame({col: np.concatenate(seq) for col, seq in self.df.items()}).groupby('target')
            r_per_target = grouped.apply(lambda df: pearsonr(df['preds'], df['targets'])[0]).mean()
            frl = grouped.apply(lambda df: self._first_rank_loss(df['preds'], df['targets'])).mean()
            return {
                'correlation_per_target': r_per_target,
                'first_rank_loss': frl,
            }

    def __init__(self, output_transform=lambda x: x):
        self.local_lddt = {
            'r2': R2(),
            'rmse': RootMeanSquaredError(),
            'correlation': PearsonR(),
            'correlation_per_model': Mean(),
        }
        
        self.global_lddt = {
            'r2': R2(),
            'rmse': RootMeanSquaredError(),
            'correlation': PearsonR(),
        }
        self.per_target_global_lddt_metrics = ProteinMetrics.PerTargetMetrics()

        self.global_gdtts = {
            'r2': R2(),
            'rmse': RootMeanSquaredError(),
            'correlation': PearsonR(),
        }
        self.per_target_global_gdtts_metrics = ProteinMetrics.PerTargetMetrics()

        self.figs = {
            'local_lddt': ScoreHistogram(title='Local LDDT'),
            'global_lddt': ScoreHistogram(title='Global LDDT'),
            'global_gdtts': ScoreHistogram(title='Global GDT_TS'),
        }
        
        self.metric_names = [
            *('metric/local_lddt/' + n for n in self.local_lddt),
            *('metric/global_lddt/' + n for n in self.global_lddt),
            *('metric/global_lddt/' + n for n in self.per_target_global_lddt_metrics.metric_names),
            *('metric/global_gdtts/' + n for n in self.global_gdtts),
            *('metric/global_gdtts/' + n for n in self.per_target_global_gdtts_metrics.metric_names),
        ]
        self.fig_names = ['fig/' + n for n in self.figs]
        
        super(ProteinMetrics, self).__init__(output_transform)
        
    def reset(self):
        for metric in chain(
                self.local_lddt.values(), 
                self.global_lddt.values(), (self.per_target_global_lddt_metrics,),
                self.global_gdtts.values(), (self.per_target_global_gdtts_metrics,),
                self.figs.values()
        ):
            metric.reset()

    def attach(self, engine, **kwargs):
        super(ProteinMetrics, self).attach(engine, None)

    def update(self, output: Tuple[List, List, GraphBatch, GraphBatch]):
        protein_names, model_names, preds, targets = output

        # All metrics: skip native structures
        non_native = torch.tensor(np.char.not_equal(model_names, 'native'), device=preds.node_features.device)

        # Local metrics: skip native structures and ignore residues that don't have a ground-truth score
        valid_scores = (
                torch.repeat_interleave(non_native, repeats=preds.num_nodes_by_graph) &
                torch.isfinite(targets.node_features[:, features.Output.Node.LOCAL_LDDT])
        )
        node_preds = preds.node_features[valid_scores, features.Output.Node.LOCAL_LDDT]
        node_targets = targets.node_features[valid_scores, features.Output.Node.LOCAL_LDDT]
        self.local_lddt['r2'].update((node_preds, node_targets))
        self.local_lddt['rmse'].update((node_preds, node_targets))
        self.local_lddt['correlation'].update((node_preds, node_targets))

        # Local correlation per model is a bit more complex, pandas is the easiest way to get a groupby
        # This dataframe is already constructed without native structures and NaN targets
        correlations_per_model = pd.DataFrame({
            # Use `node_index_by_graph` to uniquely identify a (protein, model) pair without using their str names
            'target_model': preds.node_index_by_graph[valid_scores].cpu(),
            'node_preds': node_preds.cpu(),
            'node_targets': node_targets.cpu()
        }).groupby('target_model').apply(lambda df: pearsonr(df['node_preds'], df['node_targets'])[0])
        self.local_lddt['correlation_per_model'].update(torch.from_numpy(correlations_per_model.values))

        # Global LDDT metrics 
        global_lddt_preds = preds.global_features[non_native, features.Output.Global.GLOBAL_LDDT]
        global_lddt_targets = targets.global_features[non_native, features.Output.Global.GLOBAL_LDDT]
        self.global_lddt['r2'].update((global_lddt_preds, global_lddt_targets))
        self.global_lddt['rmse'].update((global_lddt_preds, global_lddt_targets))
        self.global_lddt['correlation'].update((global_lddt_preds, global_lddt_targets))
        self.per_target_global_lddt_metrics.update(
            np.array(protein_names)[non_native.cpu().numpy()], global_lddt_preds, global_lddt_targets)

        # Global GDTTS metrics 
        global_gdtts_preds = preds.global_features[non_native, features.Output.Global.GLOBAL_GDTTS]
        global_gdtts_targets = targets.global_features[non_native, features.Output.Global.GLOBAL_GDTTS]
        self.global_gdtts['r2'].update((global_gdtts_preds, global_gdtts_targets))
        self.global_gdtts['rmse'].update((global_gdtts_preds, global_gdtts_targets))
        self.global_gdtts['correlation'].update((global_gdtts_preds, global_gdtts_targets))
        self.per_target_global_gdtts_metrics.update(
            np.array(protein_names)[non_native.cpu().numpy()], global_gdtts_preds, global_gdtts_targets)

        # Figures
        self.figs['local_lddt'].update((node_preds, node_targets))
        self.figs['global_lddt'].update((global_lddt_preds, global_lddt_targets))
        self.figs['global_gdtts'].update((global_gdtts_preds, global_gdtts_targets))

    def compute(self):
        metrics = {
            'local_lddt': {name: metric.compute() for name, metric in self.local_lddt.items()},
            'global_lddt': {
                **{name: metric.compute() for name, metric in self.global_lddt.items()},
                **self.per_target_global_lddt_metrics.compute()
            },
            'global_gdtts': {
                **{name: metric.compute() for name, metric in self.global_gdtts.items()},
                **self.per_target_global_gdtts_metrics.compute()
            },
        }
        figs = {
            name: metric.compute(pearson_r=metrics[name]['correlation'], r_squared=metrics[name]['r2'])
            for name, metric in self.figs.items()
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

        title = self.title if self.title is not None else ''
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
    def __init__(self,  target_feature: int, output_transform=lambda x: x):
        # Keep a dict of protein_name -> model_name -> PearsonR metric object
        self._targets: Mapping[str, Mapping[str, PearsonR]] = defaultdict(lambda: defaultdict(PearsonR))
        self._target_feature = target_feature
        super(LocalCorrelationPerModelSlow, self).__init__(output_transform)

    def reset(self):
        self._targets = defaultdict(lambda: defaultdict(PearsonR))

    def update(self, output):
        for target, model, preds, targets in zip(*output):
            if model != 'native':
                node_mask = torch.isfinite(targets.node_features[:, self._target_feature])
                self._targets[target][model].update((
                    preds.node_features[node_mask, self._target_feature], 
                    targets.node_features[node_mask, self._target_feature]
                ))

    def compute(self):
        return np.mean([model.compute() for target in self._targets.values() for model in target.values()])


class GlobalCorrelationPerTargetSlow(Metric):
    def __init__(self, target_feature: int, output_transform=lambda x: x):
        # Keep a dict of protein_name -> PearsonR metric object
        self._targets: Mapping[str, PearsonR] = defaultdict(PearsonR)
        self._target_feature = target_feature
        super(GlobalCorrelationPerTargetSlow, self).__init__(output_transform)

    def reset(self):
        self._targets = defaultdict(PearsonR)

    def update(self, output):
        for target, model, preds, targets in zip(*output):
            if model != 'native':
                # PearsonR expects these tensor to contain more than one value, so it needs to be a 1D tensor of len 1
                self._targets[target].update((
                    preds.global_features[self._target_feature].unsqueeze(dim=0),
                    targets.global_features[self._target_feature].unsqueeze(dim=0)
                ))

    def compute(self):
        return sum(target.compute() for target in self._targets.values()) / len(self._targets)


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
    from sklearn.metrics import mean_squared_error, r2_score

    from proteins.dataset import ProteinFolder

    metrics = ProteinMetrics()

    local_lddt_correlation_per_model = LocalCorrelationPerModelSlow(features.Output.Node.LOCAL_LDDT)
    global_lddt_correlation_per_target = GlobalCorrelationPerTargetSlow(features.Output.Global.GLOBAL_LDDT)
    global_gdtts_correlation_per_target = GlobalCorrelationPerTargetSlow(features.Output.Global.GLOBAL_GDTTS)

    dataloader_val = torch.utils.data.DataLoader(
        ProteinFolder(Path(os.environ.get('DATA_FOLDER', './data')) / 'validation', cutoff=8),
        shuffle=True, batch_size=100, collate_fn=tg.GraphBatch.collate
    )

    device = 'cuda'
    nodes_df = defaultdict(list)
    graphs_df = defaultdict(list)

    for target_name, model_name, graphs, targets in tqdm(dataloader_val):
        graphs = graphs.to(device)
        targets = targets.to(device)

        node_features = targets.node_features[:, :features.Output.Node.LENGTH]
        node_features = node_features + .3 * torch.randn_like(node_features)
        global_features = targets.global_features[:, :features.Output.Global.LENGTH]
        global_features = global_features + .2 * torch.randn_like(global_features)
        preds = graphs.evolve(node_features=node_features, global_features=global_features)

        metrics.update((target_name, model_name, preds, targets))
        local_lddt_correlation_per_model.update((target_name, model_name, preds, targets))
        global_lddt_correlation_per_target.update((target_name, model_name, preds, targets))
        global_gdtts_correlation_per_target.update((target_name, model_name, preds, targets))

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

    metrics, figs = metrics.compute().values()
    metrics['local_lddt']['correlation_per_model_slow'] = local_lddt_correlation_per_model.compute()
    metrics['global_lddt']['correlation_per_target_slow'] = global_lddt_correlation_per_target.compute()
    metrics['global_gdtts']['correlation_per_target_slow'] = global_gdtts_correlation_per_target.compute()

    print(pyaml.dump({'ignite': metrics}, safe=True))
    figs['local_lddt'].show()
    figs['global_lddt'].show()
    figs['global_gdtts'].show()

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

    def first_rank_loss(true, predicted):
        return true.max() - true[predicted.idxmax()]

    metrics['local_lddt']['rmse'] = nanrmse(nodes_df['LocalLddtTrue'], nodes_df['LocalLddtPred'])
    metrics['local_lddt']['r2'] = nanr2(nodes_df['LocalLddtTrue'], nodes_df['LocalLddtPred'])
    metrics['local_lddt']['correlation'] = nanpearsonr(nodes_df['LocalLddtTrue'], nodes_df['LocalLddtPred'])
    metrics['local_lddt']['correlation_per_model'] = nodes_df.groupby(['ProteinName', 'ModelName']) \
        .apply(lambda df: nanpearsonr(df['LocalLddtTrue'], df['LocalLddtPred'])) \
        .mean()

    graphs_df = pd.DataFrame({k: np.concatenate(v) for k, v in graphs_df.items()}).query('ModelName != "native"')

    metrics['global_lddt']['rmse'] = nanrmse(graphs_df['GlobalLddtTrue'], graphs_df['GlobalLddtPred'])
    metrics['global_lddt']['r2'] = nanr2(graphs_df['GlobalLddtTrue'], graphs_df['GlobalLddtPred'])
    metrics['global_lddt']['correlation'] = pearsonr(graphs_df['GlobalLddtTrue'], graphs_df['GlobalLddtPred'])[0]
    metrics['global_lddt']['correlation_per_target'] = graphs_df.groupby('ProteinName') \
        .apply(lambda df: pearsonr(df['GlobalLddtTrue'], df['GlobalLddtPred'])[0]) \
        .mean()
    metrics['global_lddt']['first_rank_loss'] = graphs_df.groupby('ProteinName') \
        .apply(lambda df: first_rank_loss(df['GlobalLddtTrue'], df['GlobalLddtPred'])) \
        .mean()

    metrics['global_gdtts']['rmse'] = nanrmse(graphs_df['GlobalGdttsTrue'], graphs_df['GlobalGdttsPred'])
    metrics['global_gdtts']['r2'] = nanr2(graphs_df['GlobalGdttsTrue'], graphs_df['GlobalGdttsPred'])
    metrics['global_gdtts']['correlation'] = pearsonr(graphs_df['GlobalGdttsTrue'], graphs_df['GlobalGdttsPred'])[0]
    metrics['global_gdtts']['correlation_per_target'] = graphs_df.groupby('ProteinName') \
        .apply(lambda df: pearsonr(df['GlobalGdttsTrue'], df['GlobalGdttsPred'])[0]) \
        .mean()
    metrics['global_gdtts']['first_rank_loss'] = graphs_df.groupby('ProteinName') \
        .apply(lambda df: first_rank_loss(df['GlobalGdttsTrue'], df['GlobalGdttsPred'])) \
        .mean()

    print(pyaml.dump({'pandas': metrics}, safe=True))

    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=100)
    ax.hist2d(nodes_df['LocalLddtTrue'], nodes_df['LocalLddtPred'], bins=np.linspace(0, 1, 100+1))
    ax.set_title(f'Local LDDT R: {metrics["local_lddt"]["correlation"]:.3f} R2: {metrics["local_lddt"]["r2"]:.3f}')
    ax.set_xlabel('True')
    ax.set_ylabel('Predicted')
    ax.plot([0, 1], [0, 1], color='red', linestyle='--')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.show()

    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=100)
    ax.hist2d(graphs_df['GlobalLddtTrue'], graphs_df['GlobalLddtPred'], bins=np.linspace(0, 1, 100+1))
    ax.set_title(f'Global LDDT R: {metrics["global_lddt"]["correlation"]:.3f} R2: {metrics["global_lddt"]["r2"]:.3f}')
    ax.set_xlabel('True')
    ax.set_ylabel('Predicted')
    ax.plot([0, 1], [0, 1], color='red', linestyle='--')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.show()

    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=100)
    ax.hist2d(graphs_df['GlobalGdttsTrue'], graphs_df['GlobalGdttsPred'], bins=np.linspace(0, 1, 100+1))
    ax.set_title(f'Global GDT_TS R: {metrics["global_gdtts"]["correlation"]:.3f} R2: {metrics["global_gdtts"]["r2"]:.3f}')
    ax.set_xlabel('True')
    ax.set_ylabel('Predicted')
    ax.plot([0, 1], [0, 1], color='red', linestyle='--')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.show()


if __name__ == '__main__':
    test()
