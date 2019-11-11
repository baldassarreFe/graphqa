from typing import Tuple, Union, Optional

import scipy.stats
import sklearn.metrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import ignite
from ignite.engine import Events
from ignite.metrics import Metric, RootMeanSquaredError

from .data import DecoyBatch
from .utils import rank_loss
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

    def update(self, batch: DecoyBatch):
        # Skip native structures and ignore residues that don't have a ground-truth score
        non_native = np.repeat(np.char.not_equal(batch.decoy_name, 'native'),
                               repeats=batch.num_nodes_by_graph.cpu().numpy())
        has_score = torch.isfinite(batch.lddt).cpu().numpy()
        valid_scores = np.logical_and(non_native, has_score)

        # Used to uniquely identify a (protein, model) pair without using their str names
        target_model_id = batch.node_index_by_graph[valid_scores].cpu().numpy()
        node_preds = batch.node_features[valid_scores, self.column].detach().cpu().numpy()
        node_targets = batch.lddt[valid_scores].detach().cpu().numpy()

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
            extra_title = []
            if 'pearson' in self.metrics:
                extra_title.append(f'$R$        {metrics["pearson"]:.3f}')
            if 'per_model_pearson' in self.metrics:
                extra_title.append(f'$R_\\mathrm{{model}}$ {metrics["per_model_pearson"]:.3f}')
            figures['hist'] = self._hist.compute('\n'.join(extra_title))

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
        'ranking',
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
        'paged_funnels',
        'recall_at_k',
        'ndcg_at_k',
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

    def update(self, batch: DecoyBatch):
        # Skip native structures
        non_native = np.char.not_equal(batch.decoy_name, 'native')

        protein_names = np.array(batch.target_name)[non_native]
        preds = batch.global_features[non_native, self.column].detach().cpu().numpy()
        true = batch.gdtts[non_native].detach().cpu().numpy()

        self._lists['target'].append(protein_names)
        self._lists['preds'].append(preds)
        self._lists['true'].append(true)

    @staticmethod
    def _funnel(grouped, metric, ncols):
        ncols = min(len(grouped), ncols)
        nrows = int(np.ceil(len(grouped) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows), dpi=100, squeeze=False,
                                 gridspec_kw={'hspace': 0.1, 'wspace': 0.005})

        for ax, (target_name, group) in zip(axes.ravel(), grouped):
            best_true = group['true'].idxmax()
            best_preds = group['preds'].idxmax()
            ax.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=.5, zorder=1)
            ax.scatter(group['true'], group['preds'], s=20, marker='.', zorder=2)
            ax.scatter(group['true'][best_preds], group['preds'][best_preds],
                       color='r', marker='.', zorder=3, label='Predicted')
            ax.axvline(group['true'][best_preds], color='r', linewidth=1., zorder=3)
            ax.scatter(group['true'][best_true], group['preds'][best_true],
                       color='g', marker='.', zorder=4, label='True')
            ax.axvline(group['true'][best_true], color='g', linewidth=1., zorder=4)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(target_name, fontdict={'fontsize': 'small'}, pad=1)

        axes[-1, 0].set_xticks([0, .2, .4, .6, .8, 1.])
        axes[-1, 0].set_xticklabels(['0', '', '', '', '', '1'])
        axes[-1, 0].set_yticks([0, .2, .4, .6, .8, 1.])
        axes[-1, 0].set_yticklabels(['0', '', '', '', '', '1'])
        axes[-1, 0].set_xlabel(f'True', labelpad=-10)
        axes[-1, 0].set_ylabel('Predicted', labelpad=-8)
        legend = axes[-1, 0].legend(
            title='Best decoy',
            loc='upper left',
            # title_fontsize='x-small',
            # fontsize='x-small',
            frameon=False,
            handlelength=1.,
            handletextpad=.5
        )
        legend._legend_box.align = 'left'

        # Hide unused axes
        for ax in axes.ravel()[len(grouped):]:
            ax.set_visible(False)

        # fig.show()
        # import os
        # fig.savefig(os.path.expanduser(f'~/Desktop/funnel.pdf'), bbox_inches='tight', pad_inches=0.01)
        # fig.savefig(os.path.expanduser(f'~/Desktop/funnel.png'), bbox_inches='tight', pad_inches=0.01, dpi=300)

        return fig

    @staticmethod
    def _recall_at_k(grouped, max_k, title):
        recall_at_k = grouped \
            .apply(lambda g: g.sort_values('preds', ascending=False)['true'].values.argmax() + 1) \
            .value_counts() \
            .sort_index() \
            .reindex(np.arange(1, max_k + 1), fill_value=0) \
            .cumsum() / len(grouped)

        fig, ax = plt.subplots(1, 1, dpi=100)
        # ax.step(recall_at_k.index.values, recall_at_k.values, where='post', alpha=.5)
        ax.plot(recall_at_k.index.values, recall_at_k.values, alpha=.3)
        ax.scatter(recall_at_k.index.values, recall_at_k.values, marker='.')
        ax.set_xlabel('k')
        ax.set_xticks(np.arange(0, max_k + 1, step=5 if max_k >= 10 else 1))
        ax.set_yticklabels([f'{t:.0%}' for t in ax.get_yticks()])
        ax.set_ylabel('Average Recall @ k')
        ax.set_title(title)

        return fig

    @staticmethod
    def _normalized_discounted_cumulative_gain(grouped, max_k, title):
        def max_k_normalized_dcg(group, max_k):
            discounts = np.log2(np.arange(2, min(len(group), max_k) + 2))
            dcg = np.cumsum((2 ** group.nlargest(max_k, 'preds')['true'].values - 1) / discounts)
            ideal_dcg = np.cumsum((2 ** group.nlargest(max_k, 'true')['true'].values - 1) / discounts)
            ndcg = dcg / ideal_dcg
            return pd.DataFrame(ndcg[None, :], columns=pd.RangeIndex(1, len(ndcg) + 1))

        ave_ndcg = grouped.apply(lambda group: max_k_normalized_dcg(group, max_k=max_k)).mean(skipna=True)

        fig, ax = plt.subplots(1, 1, dpi=100)
        # ax.step(ave_ndcg.index.values, ave_ndcg.values, where='post', alpha=.5)
        ax.plot(ave_ndcg.index.values, ave_ndcg.values, alpha=.3)
        ax.scatter(ave_ndcg.index.values, ave_ndcg.values, marker='.')
        ax.set_xlabel('k')
        ax.set_xticks(ave_ndcg.index.values)
        ax.set_ylabel('Average nDCG @ k')
        ax.set_title(title)

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

        if 'ranking' in self.metrics:
            metrics['ranking'] = grouped.apply(lambda g: rank_loss(g['true'], g['preds']).mean()).mean()
        if 'per_target_pearson' in self.metrics:
            metrics['per_target_pearson'] = grouped.apply(lambda g: pearson(g['preds'], g['true'])).mean()
        if 'per_target_spearman' in self.metrics:
            metrics['per_target_spearman'] = grouped.apply(lambda g: spearmanr(g['preds'], g['true'])).mean()
        if 'per_target_kendall' in self.metrics:
            metrics['per_target_kendall'] = grouped.apply(lambda g: kendalltau(g['preds'], g['true'])).mean()
        if 'first_rank_loss' in self.metrics:
            metrics['first_rank_loss'] = grouped.apply(lambda g: first_rank_loss(g['preds'], g['true'])).mean()

        if 'hist' in self.figures:
            extra_title = []
            if 'pearson' in self.metrics:
                extra_title.append(f'$R$:        {metrics["pearson"]:.3f}')
            if 'per_target_pearson' in self.metrics:
                extra_title.append(f'$R_\\mathrm{{target}}$: {metrics["per_target_pearson"]:.3f}')
            figures['hist'] = self._hist.update(df['preds'], df['true']).compute('\n'.join(extra_title))

        if 'funnel' in self.figures:
            figures['funnel'] = self._funnel(grouped, metric=self.title, ncols=6)
        if 'paged_funnels' in self.figures:
            # Funnels are drawn in pages of 6 columns and 8 rows each
            npages = int(np.ceil(len(grouped) / (6 * 8)))
            grouped_list = [(target_name, target_df) for target_name, target_df in grouped]
            for i in range(npages):
                grouped_page = grouped_list[i * (6 * 8): (i + 1) * (6 * 8)]
                figures[f'funnel_{i}'] = self._funnel(grouped_page, metric=self.title, ncols=6)
        if 'recall_at_k' in self.figures:
            figures['recall_at_k'] = self._recall_at_k(grouped, max_k=25, title=self.title)
        if 'ndcg_at_k' in self.figures:
            figures['ndcg_at_k'] = self._normalized_discounted_cumulative_gain(grouped, max_k=10, title=self.title)

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

    def compute(self, extra_title: Optional[str]):
        fig, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=100)
        ax.pcolormesh(self.bins, self.bins, self.hist.T, zorder=1)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.set_xticks([0, .2, .4, .6, .8, 1.])
        ax.set_xticklabels(['0', '', '', '', '', '1'])
        ax.set_title(self.title)
        ax.set_xlabel(f'True', labelpad=-10)
        ax.set_ylabel('Predicted')
        ax.set_yticks([0, .2, .4, .6, .8, 1.])
        ax.set_yticklabels(['0', '', '', '', '', '1'])
        ax.set_ylabel('Predicted', labelpad=-8)
        ax.plot([0, 1], [0, 1], color='red', linestyle='--', linewidth=.5, zorder=2)
        if extra_title is not None and len(extra_title) > 0:
            ax.text(
                .04, .96,
                extra_title,
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes,
                bbox={
                    'boxstyle': 'round',
                    'facecolor': 'white',
                },
            )

        # fig.show()
        # import os
        # fig.savefig(os.path.expanduser(f'~/Desktop/{self.title}_hist.pdf'), bbox_inches='tight', pad_inches=0.01)
        # fig.savefig(os.path.expanduser(f'~/Desktop/{self.title}_hist.png'), bbox_inches='tight', pad_inches=0.01, dpi=300)

        return fig


# noinspection PyAttributeOutsideInit
class ProteinAverageLosses(Metric):
    losses_names = ['local_lddt', 'global_gdtts', 'ranking']

    def reset(self):
        self._num_samples = 0
        self._loss_local_lddt = 0
        self._loss_global_gdtts = 0
        self._loss_ranking = 0

    def attach(self, engine, **ignored_kwargs):
        engine.add_event_handler(Events.EPOCH_COMPLETED, self.completed)
        if not engine.has_event_handler(self.started, Events.EPOCH_STARTED):
            engine.add_event_handler(Events.EPOCH_STARTED, self.started)
        if not engine.has_event_handler(self.iteration_completed, Events.ITERATION_COMPLETED):
            engine.add_event_handler(Events.ITERATION_COMPLETED, self.iteration_completed)

    def update(self, output: Tuple[float, float, float, int]):
        loss_local_lddt, loss_global_gdtts, loss_ranking, samples = output
        self._loss_local_lddt = combine_means(self._loss_local_lddt, loss_local_lddt, self._num_samples, samples)
        self._loss_global_gdtts = combine_means(self._loss_global_gdtts, loss_global_gdtts, self._num_samples, samples)
        self._loss_ranking = combine_means(self._loss_ranking, loss_ranking, self._num_samples, samples)
        self._num_samples += samples

    def compute(self):
        return {
            'local_lddt': self._loss_local_lddt,
            'global_gdtts': self._loss_global_gdtts,
            'ranking': self._loss_ranking,
        }

    def completed(self, engine, **ignored_kwargs):
        result = self.compute()
        engine.state.losses.update(result)
