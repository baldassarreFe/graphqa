import os
import tqdm
import pyaml
import random
import inspect
import textwrap
import namesgenerator
import multiprocessing

from typing import Mapping
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr
from munch import DefaultFactoryMunch

import torch
import torch.utils.data
import torch.nn.functional as F
import torchgraphs as tg
from torch.utils.tensorboard import SummaryWriter

from .config import parse_args
from .saver import Saver
from .utils import git_info, cuda_info, set_seeds, import_, sort_dict, RunningStats
from .dataset import ProteinFolder

# region Arguments parsing
ex = parse_args(config={
    # Experiment defaults
    'name': 'experiment',
    'fullname': '{tags}_{rand}',
    'tags': [],
    'data': {},
    'model': {},
    'optimizer': {},
    'losses': {
        'nodes': {},
        'globals': {},
    },
    'metrics': {},
    'history': [],

    # Session defaults
    'session': {
        'max_epochs': 1,
        'batch_size': 1,
        'seed': random.randint(0, 99),
        'cpus': multiprocessing.cpu_count() - 1,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'log': [],
        'checkpoint': [],
    }
})
session: dict = ex.pop('session')


# Experiment: checks and computed fields
def validate_history(history):
    if not isinstance(history, list):
        raise ValueError('History must be a list')
    for session in history:
        if session['status'] == 'FAILED':
            raise ValueError(f'Failed session in history:\n{session}')
        if session['status'] == 'COMPLETED':
            if 'samples' not in session:
                raise ValueError(f'Missing number of samples in completed session:\n{session}')
            if 'completed_epochs' not in session:
                raise ValueError(f'Missing number of epochs in completed session:\n{session}')
            
            
def validate_losses(losses):
    if not isinstance(losses, dict):
        raise ValueError('Losses must be a dict')
    for loss in losses.values():
        if 'name' not in loss:
            raise ValueError(f'Loss without a name {loss}')
        if 'weight' not in loss:
            raise ValueError(f'Loss without a weight {loss}')


if ex['name'] is None or len(ex['name']) == 0:
    raise ValueError(f'Experiment name is empty: {ex["name"]}')
if ex['tags'] is None:
    raise ValueError('Experiment tags is None')
if ex['model']['fn'] is None:
    raise ValueError('Model constructor function not defined')
if ex['optimizer']['fn'] is None:
    raise ValueError('Optimizer constructor function not defined')
validate_history(ex['history'])
validate_losses(ex['losses'])

ex['completed_epochs'] = sum((session['completed_epochs'] for session in ex['history']), 0)
ex['samples'] = sum((session['samples'] for session in ex['history']), 0)
ex['fullname'] = ex['fullname'].format(tags='_'.join(ex['tags']), rand=namesgenerator.get_random_name())

# Session computed fields
# ex['history'].append(session)  # do it here or when all epochs are done?

session['completed_epochs'] = 0
session['samples'] = 0
session['status'] = 'NEW'
session['datetime_started'] = None
session['datetime_completed'] = None
session['git'] = git_info()
session['cuda'] = cuda_info() if 'cuda' in session['device'] else None
session['metrics'] = {}

if session['cpus'] < 0:
    raise ValueError(f'Invalid number of cpus: {session["cpus"]}')
if session['seed'] is None:
    raise ValueError(f'Invalid seed: {session["seed"]}')

ex['history'].append(session)

# Print config so far
sort_dict(ex, ['name', 'tags', 'fullname', 'completed_epochs', 'samples', 'data', 'model',
               'optimizer', 'losses', 'metrics', 'history'])
sort_dict(session, ['completed_epochs', 'samples', 'max_epochs', 'batch_size', 'seed', 'cpus', 'device', 'status',
                    'datetime_started', 'datetime_completed', 'log', 'checkpoint', 'metrics', 'git', 'gpus'])

pyaml.pprint(ex, safe=True, sort_dicts=False, force_embed=True, width=200)
# endregion

# region Building phase
# Random seeds (set them after the random run id is generated)
set_seeds(session['seed'])

# Saver
saver = Saver(Path(os.environ.get('RUN_FOLDER', './runs')).joinpath(ex['fullname']))
if ex['completed_epochs'] == 0:
    saver.save_experiment(ex, epoch=ex['completed_epochs'], samples=ex['samples'])


# Model and optimizer
def load_model(config: Mapping) -> torch.nn.Module:
    special_keys = {'fn', 'state_dict'}

    if 'fn' not in config:
        raise ValueError('Model function not specified')

    function = import_(config['fn'])
    function_args = inspect.signature(function).parameters.keys()
    if not set.isdisjoint(special_keys, function_args):
        raise ValueError(f'Model function can not have any of {special_keys} in its arguments, '
                         f'signature is {", ".join(function_args)}')

    kwargs = {k: v for k, v in config.items() if k not in special_keys}
    model = function(**kwargs)

    return model


def load_optimizer(config: Mapping, model: torch.nn.Module) -> torch.optim.Optimizer:
    special_keys = {'fn'}

    if 'fn' not in config:
        raise ValueError('Optimizer function not specified')

    function = import_(config['fn'])
    function_args = inspect.signature(function).parameters.keys()
    if not set.isdisjoint(special_keys, function_args):
        raise ValueError(f'Optimizer function can not have any of {special_keys} in its arguments, '
                         f'signature is {", ".join(function_args)}')

    kwargs = {k: v for k, v in config.items() if k not in special_keys}
    optimizer = function(model.parameters(), **kwargs)

    return optimizer


model = load_model(ex['model']).to(session['device'])
optimizer = load_optimizer(ex['optimizer'], model)
if ex['completed_epochs'] > 0:  # Load latest weights and optimizer state
    model.load_state_dict(torch.load(
        saver.base_folder / 'model.latest.pt', map_location=session['device']))
    optimizer.load_state_dict(torch.load(
        saver.base_folder / 'optimizer.latest.pt', map_location=session['device']))
else:
    if 'state_dict' in ex['model']:  # A new experiment but using pretrained weights
        model.load_state_dict(torch.load(
            Path(ex['model']['state_dict']).expanduser(), map_location=session['device']))

# Logger
logger = SummaryWriter(saver.base_folder)
logger.add_text('Experiment',
                textwrap.indent(pyaml.dump(ex, safe=True, sort_dicts=False, force_embed=True), '    '),
                global_step=ex['samples'])

print('Parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))
if ex['samples'] == 0:
    logger.add_scalar(
        'misc/parameters', sum(p.numel() for p in model.parameters() if p.requires_grad), global_step=ex['samples'])
# endregion


# Datasets and dataloaders
def get_dataloaders(ex, session):
    data_folder = Path(os.environ.get('DATA_FOLDER', './data'))

    dataset_train = ProteinFolder(data_folder / 'training', cutoff=ex['data']['cutoff'])
    dataset_val = ProteinFolder(data_folder / 'validation', cutoff=ex['data']['cutoff'])

    dataloader_kwargs = dict(
        num_workers=session['cpus'],
        pin_memory='cuda' in session['device'],
        worker_init_fn=lambda _: np.random.seed(int(torch.initial_seed()) % (2 ** 32 - 1)),
        batch_size=session['batch_size'],
        collate_fn=tg.GraphBatch.collate,
    )

    dataloader_train = torch.utils.data.DataLoader(dataset_train, shuffle=True, **dataloader_kwargs)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, shuffle=False, **dataloader_kwargs)

    return dataloader_train, dataloader_val


dataloader_train, dataloader_val = get_dataloaders(ex, session)

# region Training
# Train and validation loops
session['status'] = 'RUNNING'
session['datetime_started'] = datetime.utcnow()

graphs_df = DefaultFactoryMunch(list)
nodes_df = DefaultFactoryMunch(list)

epoch_bar_postfix = {}
epoch_bar = tqdm.trange(1, session['max_epochs'] + 1, desc='Epochs', unit='e', leave=True)
for epoch_idx in epoch_bar:
    # region Training loop
    model.train()
    torch.set_grad_enabled(True)

    train_bar_postfix = {}
    loss_nodes_avg = RunningStats()
    loss_global_avg = RunningStats()
    loss_total_avg = RunningStats()

    train_bar = tqdm.tqdm(desc=f'Train {epoch_idx}', total=len(dataloader_train.dataset), unit='g')
    for batch_idx, (_, _, graphs, targets)in enumerate(dataloader_train):
        graphs = graphs.to(session['device'])
        targets = targets.to(session['device'])
        results = model(graphs)

        loss_total = torch.tensor(0., device=session['device'])

        if ex['losses']['nodes']['weight'] > 0:
            node_mask = torch.isfinite(targets.node_features[:, 0])
            loss_nodes = (
                targets.node_features[node_mask, 2] *
                F.mse_loss(results.node_features[node_mask, 0], targets.node_features[node_mask, 0], reduction='none')
            ).mean()
            loss_total += ex['losses']['nodes']['weight'] * loss_nodes
            loss_nodes_avg.add(loss_nodes.item(), len(graphs))
            train_bar_postfix['Nodes'] = f'{loss_nodes.item():.5f}'
            if 'every batch' in session['log']:
                logger.add_scalar('loss/train/nodes', loss_nodes.item(), global_step=ex['samples'])

        if ex['losses']['globals']['weight'] > 0:
            loss_global = F.mse_loss(
                results.global_features.squeeze(), targets.global_features.squeeze(), reduction='mean')
            loss_total += ex['losses']['globals']['weight'] * loss_global
            loss_global_avg.add(loss_global.item(), len(graphs))
            train_bar_postfix['Global'] = f'{loss_global.item():.5f}'
            if 'every batch' in session['log']:
                logger.add_scalar('loss/train/global', loss_global.mean().item(), global_step=ex["samples"])

        loss_total_avg.add(loss_total.item(), len(graphs))
        train_bar_postfix['Total'] = f'{loss_total.item():.5f}'
        if 'every batch' in session['log']:
            logger.add_scalar('loss/train/total', loss_total.item(), global_step=ex["samples"])

        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step(closure=None)

        ex['samples'] += len(graphs)
        session['samples'] += len(graphs)
        train_bar.update(len(graphs))
        train_bar.set_postfix(train_bar_postfix)

        if 'QUICK_RUN' in os.environ and batch_idx >= 5:
            train_bar.write(f'Interrupting training loop after {batch_idx} batches')
            break
    train_bar.close()

    epoch_bar_postfix['Train'] = f'{loss_total_avg.mean:.4f}'
    epoch_bar.set_postfix(epoch_bar_postfix)

    if 'every epoch' in session['log'] and 'every batch' not in session['log']:
        logger.add_scalar('loss/train/total', loss_total_avg.mean, global_step=ex['samples'])
        if session['losses']['nodes'] > 0:
            logger.add_scalar('loss/train/nodes', loss_nodes_avg.mean, global_step=ex['samples'])
        if session['losses']['count'] > 0:
            logger.add_scalar('loss/train/global', loss_global_avg.mean, global_step=ex['samples'])

    del batch_idx, train_bar, train_bar_postfix, loss_nodes_avg, loss_global_avg, loss_total_avg
    # endregion

    # region Validation loop
    model.eval()
    torch.set_grad_enabled(False)

    val_bar_postfix = {}
    loss_nodes_avg = RunningStats()
    loss_global_avg = RunningStats()
    loss_total_avg = RunningStats()

    val_bar = tqdm.tqdm(desc=f'Val {epoch_idx}', total=len(dataloader_val.dataset), unit='g')
    for batch_idx, (protein_name, model_name, graphs, targets) in enumerate(dataloader_val):
        graphs = graphs.to(session['device'])
        targets = targets.to(session['device'])
        results = model(graphs)

        loss_total = torch.tensor(0., device=session['device'])

        if ex['losses']['nodes']['weight'] > 0:
            node_mask = torch.isfinite(targets.node_features[:, 0])
            loss_nodes = (
                targets.node_features[node_mask, 2] *
                F.mse_loss(results.node_features[node_mask, 0], targets.node_features[node_mask, 0], reduction='none')
            ).mean()
            loss_total += ex['losses']['nodes']['weight'] * loss_nodes
            loss_nodes_avg.add(loss_nodes.item(), len(graphs))
            val_bar_postfix['Nodes'] = f'{loss_nodes.item():.5f}'

        if ex['losses']['globals']['weight'] > 0:
            loss_global = F.mse_loss(
                results.global_features.squeeze(), targets.global_features.squeeze(), reduction='mean')
            loss_total += ex['losses']['globals']['weight'] * loss_global
            loss_global_avg.add(loss_global.item(), len(graphs))
            val_bar_postfix['Global'] = f'{loss_global.item():.5f}'

        val_bar_postfix['Total'] = f'{loss_total.item():.5f}'
        loss_total_avg.add(loss_total.item(), len(graphs))

        # region Last epoch
        if epoch_idx == session['max_epochs']:
            import torch_scatter
            # Drop residues not present in the native structure
            present_in_native = targets.node_features[:, 1] == 1.
            global_indices = graphs.node_index_by_graph[present_in_native]

            # Set to 0 the score of the residues not present in the model
            local_scores = results.node_features[present_in_native, 0]
            missing_in_model = graphs.node_features[present_in_native, 82] == 0.
            local_scores[missing_in_model] = 0.

            global_scores = torch_scatter.scatter_mean(local_scores, index=global_indices, dim=0, dim_size=graphs.num_graphs)

            graphs_df['ProteinName'].append(np.array(protein_name))
            graphs_df['ModelName'].append(np.array(model_name))
            graphs_df['GlobalScoreTrue'].append(targets.global_features.squeeze(dim=1).cpu())
            graphs_df['GlobalScoreComputed'].append(global_scores.cpu())
            graphs_df['GlobalScorePredicted'].append(results.global_features.squeeze(dim=1).cpu())

            nodes_df['ProteinName'].append(np.repeat(np.array(protein_name), graphs.num_nodes_by_graph.cpu()))
            nodes_df['ModelName'].append(np.repeat(np.array(model_name), graphs.num_nodes_by_graph.cpu()))
            nodes_df['NodeId'].append(np.concatenate([np.arange(n) for n in graphs.num_nodes_by_graph.cpu()]))
            nodes_df['LocalScoreTrue'].append(targets.node_features[:, 0].cpu())
            nodes_df['LocalScorePredicted'].append(results.node_features.squeeze().cpu())
        # endregion

        val_bar.update(len(graphs))
        val_bar.set_postfix(val_bar_postfix)

        if 'QUICK_RUN' in os.environ and batch_idx >= 5:
            val_bar.write(f'Interrupting validation loop after {batch_idx} batches')
            break
    val_bar.close()

    ex['completed_epochs'] += 1
    session['completed_epochs'] += 1
    epoch_bar_postfix['Val'] = f'{loss_total_avg.mean:.4f}'
    epoch_bar.set_postfix(epoch_bar_postfix)

    if (
            'every batch' in session['log'] or
            'every epoch' in session['log'] or
            'last epoch' in session['log'] and epoch_idx == session['max_epochs']
    ):
        logger.add_scalar('loss/val/total', loss_total_avg.mean, global_step=ex['samples'])
        if ex['losses']['nodes']['weight'] > 0:
            logger.add_scalar('loss/val/nodes', loss_nodes_avg.mean, global_step=ex['samples'])
        if ex['losses']['globals']['weight'] > 0:
            logger.add_scalar('loss/val/global', loss_global_avg.mean, global_step=ex['samples'])

    del batch_idx, val_bar, val_bar_postfix, loss_nodes_avg, loss_global_avg, loss_total_avg
    # endregion

    # Saving
    if epoch_idx == session['max_epochs']:
        session['status'] = 'DONE'
        session['datetime_completed'] = datetime.utcnow()
    if (
            'every batch' in session['checkpoint'] or
            'every epoch' in session['checkpoint'] or
            ('last epoch' in session['checkpoint'] and epoch_idx == session['max_epochs'])
    ):
        saver.save(model, ex, optimizer, epoch=ex['completed_epochs'], samples=ex['samples'])
epoch_bar.close()
print()
del epoch_bar, epoch_bar_postfix, epoch_idx
# endregion

# region Final report
if 'cuda' in session['device']:
    for device_id, device_info in session['cuda']['devices'].items():
        device_info.update({
            'memory_used_max': f'{torch.cuda.max_memory_allocated(device_id) // (10**6)} MiB',
            'memory_cached_max': f'{torch.cuda.max_memory_cached(device_id) // (10**6)} MiB',
        })
    print('GPU usage:', pyaml.dump(session['cuda']['devices'], safe=True, sort_dicts=False), sep='\n')

metrics = ex['metrics'] = session['metrics'] = {'local': {}, 'globals': {}, 'globals_computed': {}}
graphs_df = pd.DataFrame({k: np.concatenate(v) for k, v in graphs_df.items()})

metrics['globals']['rmse'] = np.sqrt(mean_squared_error(graphs_df['GlobalScoreTrue'], graphs_df['GlobalScorePredicted']))
metrics['globals']['R2'] = r2_score(graphs_df['GlobalScoreTrue'], graphs_df['GlobalScorePredicted'])
metrics['globals']['R2_per_target'] = graphs_df.groupby('ProteinName') \
    .apply(lambda df: r2_score(df['GlobalScoreTrue'], df['GlobalScorePredicted'])) \
    .mean()
metrics['globals']['pearson_R'] = pearsonr(graphs_df['GlobalScoreTrue'], graphs_df['GlobalScorePredicted'])[0]
metrics['globals']['pearson_R_per_target'] = graphs_df.groupby('ProteinName') \
    .apply(lambda df: pearsonr(df['GlobalScoreTrue'], df['GlobalScorePredicted'])[0]) \
    .mean()

metrics['globals_computed']['rmse'] = np.sqrt(mean_squared_error(graphs_df['GlobalScoreTrue'], graphs_df['GlobalScoreComputed']))
metrics['globals_computed']['R2'] = r2_score(graphs_df['GlobalScoreTrue'], graphs_df['GlobalScoreComputed'])
metrics['globals_computed']['R2_per_target'] = graphs_df.groupby('ProteinName') \
    .apply(lambda df: r2_score(df['GlobalScoreTrue'], df['GlobalScoreComputed'])) \
    .mean()
metrics['globals_computed']['pearson_R'] = pearsonr(graphs_df['GlobalScoreTrue'], graphs_df['GlobalScoreComputed'])[0]
metrics['globals_computed']['pearson_R_per_target'] = graphs_df.groupby('ProteinName') \
    .apply(lambda df: pearsonr(df['GlobalScoreTrue'], df['GlobalScoreComputed'])[0]) \
    .mean()

nodes_df = pd.DataFrame({k: np.concatenate(v) for k, v in nodes_df.items()}).dropna(subset=['LocalScoreTrue'])
metrics['local']['rmse'] = np.sqrt(mean_squared_error(nodes_df['LocalScoreTrue'], nodes_df['LocalScorePredicted']))
metrics['local']['R2'] = r2_score(nodes_df['LocalScoreTrue'], nodes_df['LocalScorePredicted'])
metrics['local']['R2_per_model'] = nodes_df.groupby(['ProteinName', 'ModelName']) \
    .apply(lambda df: r2_score(df['LocalScoreTrue'], df['LocalScorePredicted'])) \
    .mean()
metrics['local']['pearson_R'] = pearsonr(nodes_df['LocalScoreTrue'], nodes_df['LocalScorePredicted'])[0]
metrics['local']['pearson_R_per_model'] = nodes_df.groupby(['ProteinName', 'ModelName']) \
    .apply(lambda df: pearsonr(df['LocalScoreTrue'], df['LocalScorePredicted'])[0]) \
    .mean()

pyaml.pprint(metrics, safe=True, sort_dicts=False, force_embed=True, width=200)
for category in metrics:
    for name, value in metrics[category].items():
        logger.add_scalar(f'metric/{category}/{name}', value, global_step=ex['samples'])


def plot_metrics(nodes_df, graphs_df, metrics):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=100)

    ax = axes[0]
    ax.hist2d(nodes_df['LocalScoreTrue'], nodes_df['LocalScorePredicted'], bins=np.linspace(0, 1, 100+1))
    ax.set_title(f'Local Scores (R: {metrics["local"]["pearson_R"]:.3f} R2: {metrics["local"]["R2"]:.3f})')
    ax.set_xlabel('True')
    ax.set_ylabel('Predicted')
    ax.plot([0, 1], [0, 1], color='red', linestyle='--')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax = axes[1]
    ax.hist2d(graphs_df['GlobalScoreTrue'], graphs_df['GlobalScorePredicted'], bins=np.linspace(0, 1, 100+1))
    ax.set_title(f'Global Scores (R: {metrics["globals"]["pearson_R"]:.3f} R2: {metrics["globals"]["R2"]:.3f})')
    ax.set_xlabel('True')
    ax.set_ylabel('Predicted')
    ax.plot([0, 1], [0, 1], color='red', linestyle='--')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    return fig


logger.add_figure('metrics', plot_metrics(nodes_df, graphs_df, metrics), close=True, global_step=ex['samples'])

logger.add_text('Experiment',
                textwrap.indent(pyaml.dump(ex, safe=True, sort_dicts=False, force_embed=True), '    '),
                ex['samples'])

del graphs_df, nodes_df
# endregion

# region Cleanup
saver.save_experiment(ex, epoch=ex['completed_epochs'], samples=ex['samples'])
logger.close()
# endregion
