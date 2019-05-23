import tqdm
import yaml
import pyaml
import random
import textwrap
import multiprocessing

from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr
from munch import AutoMunch, DefaultFactoryMunch

import torch
import torch.utils.data
import torch.nn.functional as F
import torchgraphs as tg
from torch.utils.tensorboard import SummaryWriter

from .saver import Saver
from .utils import git_info, cuda_info, parse_dotted, update_rec, set_seeds, import_, sort_dict, RunningStats
from .dataset import ProteinFile

parser = ArgumentParser()
parser.add_argument('--experiment', nargs='+', required=True)
parser.add_argument('--model', nargs='+', required=False, default=[])
parser.add_argument('--optimizer', nargs='+', required=False, default=[])
parser.add_argument('--session', nargs='+', required=False, default=[])

args = parser.parse_args()


# region Collecting phase
class Experiment(AutoMunch):
    @property
    def session(self):
        return self.sessions[-1]


experiment = Experiment()

# Experiment defaults
experiment.name = 'experiment'
experiment.tags = []
experiment.samples = 0
experiment.model = {'fn': None, 'args': [], 'kwargs': {}}
experiment.optimizer = {'fn': None, 'args': [], 'kwargs': {}}
experiment.sessions = []

# Session defaults
session = AutoMunch()
session.losses = {'nodes': 0, 'globals': 0}
session.seed = random.randint(0, 99)
session.cpus = multiprocessing.cpu_count() - 1
session.device = 'cuda' if torch.cuda.is_available() else 'cpu'
session.log = {'when': []}
session.checkpoint = {'when': []}

# Experiment configuration
for string in args.experiment:
    if '=' in string:
        update = parse_dotted(string)
    else:
        with open(string, 'r') as f:
            update = yaml.safe_load(f)
    # If the current session is defined inside the experiment update the session instead
    if 'session' in update:
        update_rec(session, update.pop('session'))
    update_rec(experiment, update)

# Model from --model args
for string in args.model:
    if '=' in string:
        update = parse_dotted(string)
    else:
        with open(string, 'r') as f:
            update = yaml.safe_load(f)
            # If the yaml document contains a single entry with key `model` use that one instead
            if update.keys() == {'model'}:
                update = update['model']
    update_rec(experiment.model, update)
    del update

# Optimizer from --optimizer args
for string in args.optimizer:
    if '=' in string:
        update = parse_dotted(string)
    else:
        with open(string, 'r') as f:
            update = yaml.safe_load(f)
            # If the yaml document contains a single entry with key `optimizer` use that one instead
            if update.keys() == {'optimizer'}:
                update = update['optimizer']
    update_rec(experiment.optimizer, update)
    del update

# Session from --session args
for string in args.session:
    if '=' in string:
        update = parse_dotted(string)
    else:
        with open(string, 'r') as f:
            update = yaml.safe_load(f)
            # If the yaml document contains a single entry with key `session` use that one instead
            if update.keys() == {'session'}:
                update = update['session']
    update_rec(session, update)
    del update

# Checks (some missing, others redundant)
if experiment.name is None or len(experiment.name) == 0:
    raise ValueError(f'Experiment name is empty: {experiment.name}')
if experiment.tags is None:
    raise ValueError('Experiment tags is None')
if experiment.model.fn is None:
    raise ValueError('Model constructor function not defined')
if experiment.optimizer.fn is None:
    raise ValueError('Optimizer constructor function not defined')
if session.cpus < 0:
    raise ValueError(f'Invalid number of cpus: {session.cpus}')
if any(l < 0 for l in session.losses.values()) or all(loss == 0 for loss in session.losses.values()):
    raise ValueError(f'Invalid losses: {session.losses}')
if len(experiment.sessions) > 0 and ('state_dict' not in experiment.model or 'state_dict' not in experiment.optimizer):
    raise ValueError(f'Model and optimizer state dicts are required to restore training')

# Experiment computed fields
experiment.epoch = sum((s.epochs for s in experiment.sessions), 0)

# Session computed fields
session.status = 'NEW'
session.datetime_started = None
session.datetime_completed = None
git = git_info()
if git is not None:
    session.git = git
if 'cuda' in session.device:
    session.cuda = cuda_info()

# Resolving paths
rand_id = ''.join(chr(random.randint(ord('A'), ord('Z'))) for _ in range(6))
session.data.folder = Path(session.data.folder.replace('{name}', experiment.name)).expanduser().resolve().as_posix()
session.log.folder = session.log.folder \
    .replace('{name}', experiment.name) \
    .replace('{tags}', '_'.join(experiment.tags)) \
    .replace('{rand}', rand_id)
if len(session.checkpoint.when) > 0:
    if len(session.log.when) > 0:
        session.log.folder = Path(session.log.folder).expanduser().resolve().as_posix()
    session.checkpoint.folder = session.checkpoint.folder \
        .replace('{name}', experiment.name) \
        .replace('{tags}', '_'.join(experiment.tags)) \
        .replace('{rand}', rand_id)
    session.checkpoint.folder = Path(session.checkpoint.folder).expanduser().resolve().as_posix()
if 'state_dict' in experiment.model:
    experiment.model.state_dict = Path(experiment.model.state_dict).expanduser().resolve().as_posix()
if 'state_dict' in experiment.optimizer:
    experiment.optimizer.state_dict = Path(experiment.optimizer.state_dict).expanduser().resolve().as_posix()

sort_dict(experiment, ['name', 'tags', 'epoch', 'samples', 'model', 'optimizer', 'sessions'])
sort_dict(session, ['epochs', 'batch_size', 'losses', 'seed', 'cpus', 'device', 'samples', 'status',
                    'datetime_started', 'datetime_completed', 'data', 'log', 'checkpoint', 'git', 'gpus'])
experiment.sessions.append(session)
pyaml.pprint(experiment, sort_dicts=False, width=200)
del session
# endregion

# region Building phase
# Seeds (set them after the random run id is generated)
set_seeds(experiment.session.seed)

# Model
model: torch.nn.Module = import_(experiment.model.fn)(*experiment.model.args, **experiment.model.kwargs)
if 'state_dict' in experiment.model:
    model.load_state_dict(torch.load(experiment.model.state_dict))
model.to(experiment.session.device)

# Optimizer
optimizer: torch.optim.Optimizer = import_(experiment.optimizer.fn)(
    model.parameters(), *experiment.optimizer.args, **experiment.optimizer.kwargs)
if 'state_dict' in experiment.optimizer:
    optimizer.load_state_dict(torch.load(experiment.optimizer.state_dict))

# Logger
if len(experiment.session.log.when) > 0:
    logger = SummaryWriter(experiment.session.log.folder)
    logger.add_text(
        'Experiment', textwrap.indent(pyaml.dump(experiment, safe=True, sort_dicts=False), '    '), experiment.samples)
else:
    logger = None

# Saver
if len(experiment.session.checkpoint.when) > 0:
    saver = Saver(experiment.session.checkpoint.folder)
    if experiment.epoch == 0:
        saver.save_experiment(experiment, suffix=f'e{experiment.epoch:04d}')
else:
    saver = None

params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Parameters:', params)
if logger is not None:
    logger.add_scalar('misc/parameters', params, global_step=experiment.samples)
del params
# endregion

# Datasets and dataloaders
dataset_train = ProteinFile(Path(experiment.session.data.folder) / 'training_casp9_10.v4.h5')
dataset_val = ProteinFile(Path(experiment.session.data.folder) / 'validation_casp11.v4.h5')

dataloader_kwargs = dict(
    num_workers=min(experiment.session.cpus, 1) if 'cuda' in experiment.session.device else experiment.session.cpus,
    pin_memory='cuda' in experiment.session.device,
    worker_init_fn=lambda _: np.random.seed(int(torch.initial_seed()) % (2 ** 32 - 1)),
    batch_size=experiment.session.batch_size,
    collate_fn=tg.GraphBatch.collate,
)
dataloader_train = torch.utils.data.DataLoader(
    dataset_train,
    shuffle=True,
    **dataloader_kwargs
)
dataloader_val = torch.utils.data.DataLoader(
    dataset_val,
    shuffle=False,
    **dataloader_kwargs
)
# count_weights = pd.Series(t.global_features.item() for g, t in dataset_train) \
#     .value_counts(normalize=True) \
#     .apply(np.log) \
#     .apply(np.negative) \
#     .astype(np.float32) \
#     .sort_index()
del dataloader_kwargs


# region Training
# Train and validation loops
experiment.session.status = 'RUNNING'
experiment.session.datetime_started = datetime.utcnow()

graphs_df = DefaultFactoryMunch(list)
nodes_df = DefaultFactoryMunch(list)

epoch_bar_postfix = {}
epoch_bar = tqdm.trange(1, experiment.session.epochs + 1, desc='Epochs', unit='e', leave=True)
for epoch_idx in epoch_bar:
    experiment.epoch += 1

    # region Training loop
    model.train()
    torch.set_grad_enabled(True)

    train_bar_postfix = {}
    loss_nodes_avg = RunningStats()
    loss_global_avg = RunningStats()
    loss_total_avg = RunningStats()

    train_bar = tqdm.tqdm(desc=f'Train {experiment.epoch}', total=len(dataloader_train.dataset), unit='g')
    for batch_idx, (_, _, graphs, targets )in enumerate(dataloader_train):
        graphs = graphs.to(experiment.session.device)
        targets = targets.to(experiment.session.device)
        results = model(graphs)

        loss_total = torch.tensor(0., device=experiment.session.device)

        if experiment.session.losses.nodes > 0:
            node_mask = ~torch.isnan(targets.node_features.squeeze())
            loss_nodes = F.mse_loss(results.node_features.squeeze()[node_mask],
                                    targets.node_features.squeeze()[node_mask], reduction='mean')
            loss_total += experiment.session.losses.nodes * loss_nodes
            loss_nodes_avg.add(loss_nodes.item(), len(graphs))
            train_bar_postfix['Nodes'] = f'{loss_nodes.item():.5f}'
            if 'every batch' in experiment.session.log.when:
                logger.add_scalar('loss/train/nodes', loss_nodes.item(), global_step=experiment.samples)

        if experiment.session.losses.globals > 0:
            loss_global = F.mse_loss(
                results.global_features.squeeze(), targets.global_features.squeeze(), reduction='mean')
            loss_total += experiment.session.losses.globals * loss_global
            loss_global_avg.add(loss_global.item(), len(graphs))
            train_bar_postfix['Global'] = f'{loss_global.item():.5f}'
            if 'every batch' in experiment.session.log.when:
                logger.add_scalar('loss/train/global', loss_global.mean().item(), global_step=experiment.samples)

        loss_total_avg.add(loss_total.item(), len(graphs))
        train_bar_postfix['Total'] = f'{loss_total.item():.5f}'
        if 'every batch' in experiment.session.log.when:
            logger.add_scalar('loss/train/total', loss_total.item(), global_step=experiment.samples)

        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step(closure=None)

        experiment.samples += len(graphs)
        train_bar.update(len(graphs))
        train_bar.set_postfix(train_bar_postfix)
        # if batch_idx == 5:
        #     break
    train_bar.close()

    epoch_bar_postfix['Train'] = f'{loss_total_avg.mean:.4f}'
    epoch_bar.set_postfix(epoch_bar_postfix)

    if 'every epoch' in experiment.session.log.when and 'every batch' not in experiment.session.log.when:
        logger.add_scalar('loss/train/total', loss_total_avg.mean, global_step=experiment.samples)
        if experiment.session.losses.nodes > 0:
            logger.add_scalar('loss/train/nodes', loss_nodes_avg.mean, global_step=experiment.samples)
        if experiment.session.losses.count > 0:
            logger.add_scalar('loss/train/global', loss_global_avg.mean, global_step=experiment.samples)

    del batch_idx, train_bar, train_bar_postfix, loss_nodes_avg, loss_global_avg, loss_total_avg
    # endregion

    # region Validation loop
    model.eval()
    torch.set_grad_enabled(False)

    val_bar_postfix = {}
    loss_nodes_avg = RunningStats()
    loss_global_avg = RunningStats()
    loss_total_avg = RunningStats()

    val_bar = tqdm.tqdm(desc=f'Val {experiment.epoch}', total=len(dataloader_val.dataset), unit='g')
    for batch_idx, (protein_name, model_name, graphs, targets) in enumerate(dataloader_val):
        graphs = graphs.to(experiment.session.device)
        targets = targets.to(experiment.session.device)
        results = model(graphs)

        loss_total = torch.tensor(0., device=experiment.session.device)

        if experiment.session.losses.nodes > 0:
            node_mask = ~torch.isnan(targets.node_features.squeeze())
            loss_nodes = F.mse_loss(results.node_features.squeeze()[node_mask],
                                    targets.node_features.squeeze()[node_mask], reduction='mean')
            loss_total += experiment.session.losses.nodes * loss_nodes
            loss_nodes_avg.add(loss_nodes.item(), len(graphs))
            val_bar_postfix['Nodes'] = f'{loss_nodes.item():.5f}'

        if experiment.session.losses.globals > 0:
            loss_global = F.mse_loss(
                results.global_features.squeeze(), targets.global_features.squeeze(), reduction='mean')
            loss_total += experiment.session.losses.globals * loss_global
            loss_global_avg.add(loss_global.item(), len(graphs))
            val_bar_postfix['Global'] = f'{loss_global.item():.5f}'

        val_bar_postfix['Total'] = f'{loss_total.item():.5f}'
        loss_total_avg.add(loss_total.item(), len(graphs))

        # region Last epoch
        if epoch_idx == experiment.session.epochs:
            graphs_df['ProteinName'].append(np.array(protein_name))
            graphs_df['ModelName'].append(np.array(model_name))
            graphs_df['GlobalScoreTrue'].append(targets.global_features.squeeze().cpu())
            graphs_df['GlobalScorePredicted'].append(results.global_features.squeeze().cpu())

            nodes_df['ProteinName'].append(np.repeat(np.array(protein_name), graphs.num_nodes_by_graph.cpu()))
            nodes_df['ModelName'].append(np.repeat(np.array(model_name), graphs.num_nodes_by_graph.cpu()))
            nodes_df['NodeId'].append(np.concatenate([np.arange(n) for n in graphs.num_nodes_by_graph.cpu()]))
            nodes_df['LocalScoreTrue'].append(targets.node_features.squeeze().cpu())
            nodes_df['LocalScorePredicted'].append(results.node_features.squeeze().cpu())
        # endregion

        val_bar.update(len(graphs))
        val_bar.set_postfix(val_bar_postfix)
        # if batch_idx == 5:
        #     break
    val_bar.close()

    epoch_bar_postfix['Val'] = f'{loss_total_avg.mean:.4f}'
    epoch_bar.set_postfix(epoch_bar_postfix)

    if (
            'every batch' in experiment.session.log.when or
            'every epoch' in experiment.session.log.when or
            'last epoch' in experiment.session.checkpoint.when and epoch_idx == experiment.session.epochs
    ):
        logger.add_scalar('loss/val/total', loss_total_avg.mean, global_step=experiment.samples)
        if experiment.session.losses.nodes > 0:
            logger.add_scalar('loss/val/nodes', loss_nodes_avg.mean, global_step=experiment.samples)
        if experiment.session.losses.globals > 0:
            logger.add_scalar('loss/val/global', loss_global_avg.mean, global_step=experiment.samples)

    del batch_idx, val_bar, val_bar_postfix, loss_nodes_avg, loss_global_avg, loss_total_avg
    # endregion

    # Saving
    if epoch_idx == experiment.session.epochs:
        experiment.session.status = 'DONE'
        experiment.session.datetime_completed = datetime.utcnow()
    if (
            'every batch' in experiment.session.checkpoint.when or
            'every epoch' in experiment.session.checkpoint.when or
            'last epoch' in experiment.session.checkpoint.when and epoch_idx == experiment.session.epochs
    ):
        saver.save(model, experiment, optimizer, suffix=f'e{experiment.epoch:04d}')
epoch_bar.close()
print()
del epoch_bar, epoch_bar_postfix, epoch_idx
# endregion

# region Final report
graphs_df = pd.DataFrame({k: np.concatenate(v) for k, v in graphs_df.items()})
experiment.metrics = {'globals': {}, 'local': {}}
experiment.metrics.globals.rmse = np.sqrt(mean_squared_error(graphs_df['GlobalScoreTrue'], graphs_df['GlobalScorePredicted']))
experiment.metrics.globals.R2 = r2_score(graphs_df['GlobalScoreTrue'], graphs_df['GlobalScorePredicted'])
experiment.metrics.globals.R2_per_target = graphs_df.groupby('ProteinName') \
    .apply(lambda df: r2_score(df['GlobalScoreTrue'], df['GlobalScorePredicted'])) \
    .mean()
experiment.metrics.globals.pearson_R = pearsonr(graphs_df['GlobalScoreTrue'], graphs_df['GlobalScorePredicted'])[0]
experiment.metrics.globals.pearson_R_per_target = graphs_df.groupby('ProteinName') \
    .apply(lambda df: pearsonr(df['GlobalScoreTrue'], df['GlobalScorePredicted'])[0]) \
    .mean()

nodes_df = pd.DataFrame({k: np.concatenate(v) for k, v in nodes_df.items()}).dropna(subset=['LocalScoreTrue'])
experiment.metrics.local.rmse = np.sqrt(mean_squared_error(nodes_df['LocalScoreTrue'], nodes_df['LocalScorePredicted']))
experiment.metrics.local.R2 = r2_score(nodes_df['LocalScoreTrue'], nodes_df['LocalScorePredicted'])
experiment.metrics.local.R2_per_model = nodes_df.groupby(['ProteinName', 'ModelName']) \
    .apply(lambda df: r2_score(df['LocalScoreTrue'], df['LocalScorePredicted'])) \
    .mean()
experiment.metrics.local.pearson_R = pearsonr(nodes_df['LocalScoreTrue'], nodes_df['LocalScorePredicted'])[0]
experiment.metrics.local.pearson_R_per_model = nodes_df.groupby(['ProteinName', 'ModelName']) \
    .apply(lambda df: pearsonr(df['LocalScoreTrue'], df['LocalScorePredicted'])[0]) \
    .mean()

pyaml.pprint(experiment.metrics, sort_dicts=False, width=200)

if logger is not None:
    for sub in ['local', 'globals']:
        for name, value in experiment.metrics[sub].items():
            logger.add_scalar(f'metric/{sub}/{name}', value, global_step=experiment.samples)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=100)

    ax = axes[0]
    ax.hist2d(nodes_df['LocalScoreTrue'], nodes_df['LocalScorePredicted'], bins=np.linspace(0, 1, 100+1))
    ax.set_title(f'Local Scores (R: {experiment.metrics.local.pearson_R:.3f} R2: {experiment.metrics.local.R2:.3f})')
    ax.set_xlabel('True')
    ax.set_ylabel('Predicted')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax = axes[1]
    ax.hist2d(graphs_df['GlobalScoreTrue'], graphs_df['GlobalScorePredicted'], bins=np.linspace(0, 1, 100+1))
    ax.set_title(f'Global Scores (R: {experiment.metrics.globals.pearson_R:.3f} R2: {experiment.metrics.globals.R2:.3f})')
    ax.set_xlabel('True')
    ax.set_ylabel('Predicted')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    logger.add_figure('metrics', fig, global_step=experiment.samples, close=True)

    logger.add_text(
        'Experiment', textwrap.indent(pyaml.dump(experiment, safe=True, sort_dicts=False), '    '), experiment.samples)

# N = 100
# pd.options.display.precision = 2
# pd.options.display.max_columns = 999
# pd.options.display.expand_frame_repr = False

# # Split the results in ranges based on the number of nodes and compute the average loss per range
# df_losses_by_node_range = graphs_df \
#     .groupby(graphs_df.NumNodes // N) \
#     .agg({'NumNodes': ['min', 'max'], 'ModelName': 'count', 'LossNodes': 'mean', 'LossGlobal': 'mean'}) \
#     .rename_axis(index='NodeRange') \
#     .rename(lambda node_group_min: f'[{node_group_min * N}, {node_group_min * N + N})', axis='index') \
#     .rename(str.capitalize, axis='columns', level=1)
#
# # Split the results in ranges based on the number of nodes and keep the 5 worst predictions w.r.t. node-wise loss
# df_worst_nodes_loss_by_node_range = graphs_df \
#     .groupby(graphs_df.Nodes // N) \
#     .apply(lambda df_gr: df_gr.nlargest(5, 'LossNodes').set_index('GraphId')) \
#     .rename_axis(index={'Nodes': 'NodeRange'}) \
#     .rename(lambda node_group_min: f'[{node_group_min * N}, {node_group_min * N + N})', axis='index', level=0)
#
# # Split the results in ranges based on the number of nodes and keep the 5 worst predictions w.r.t. graph-wise loss
# df_worst_global_loss_by_node_range = graphs_df \
#     .groupby(graphs_df.Nodes // N) \
#     .apply(lambda df_gr: df_gr.nlargest(5, 'LossGlobal').set_index('GraphId')) \
#     .rename_axis(index={'Nodes': 'NodeRange'}) \
#     .rename(lambda node_group_min: f'[{node_group_min * N}, {node_group_min * N + N})', axis='index', level=0)
#
# print(f"""
# Losses by range:
# {df_losses_by_node_range}\n
# Worst node predictions:
# {df_worst_nodes_loss_by_node_range}\n
# Worst global predictions:
# {df_worst_global_loss_by_node_range}
# """)
#
# if logger is not None:
#     logger.add_text(
#         'Losses by range',
#         textwrap.indent(df_losses_by_node_range.to_string(), '    '),
#         global_step=experiment.samples)
#     logger.add_text(
#         'Worst node predictions',
#         textwrap.indent(df_worst_nodes_loss_by_node_range.to_string(), '    '),
#         global_step=experiment.samples),
#     logger.add_text(
#         'Worst global predictions',
#         textwrap.indent(df_worst_global_loss_by_node_range.to_string(), '    '),
#         global_step=experiment.samples)

del graphs_df, nodes_df
# del df_losses_by_node_range, df_worst_nodes_loss_by_node_range, df_worst_global_loss_by_node_range
# endregion

# region Cleanup
if saver is not None:
    saver.save_experiment(experiment, suffix=f'e{experiment.epoch:04d}')

if logger is not None:
    logger.close()
# endregion
