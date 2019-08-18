import os
import pyaml
import random
import inspect
import textwrap
import multiprocessing

from typing import Mapping
from pathlib import Path
from datetime import datetime
from operator import itemgetter

import numpy as np
import namesgenerator

import torch
import torch.utils.data
import torch.nn.functional as F
import torchgraphs as tg
from tensorboardX import SummaryWriter

from ignite.engine import Engine, Events
from ignite.contrib.handlers import ProgressBar

from . import features
from .config import build_dict
from .utils import round_timedelta
from .config import parse_args
from .saver import Saver
from .utils import git_info, cuda_info, set_seeds, import_, sort_dict
from .dataset import ProteinFolder, PositionalEncoding, RemoveEdges, RbfDistEdges
from .metrics import ProteinMetrics, ProteinAverageLosses
from .base_metrics import GpuMaxMemoryAllocated
from .my_hparams import make_session_start_summary, make_session_end_summary

# region Arguments parsing
ex = parse_args(config={
    # Experiment defaults
    'name': 'experiment',
    'fullname': '{tags}_{rand}',
    'tags': [],
    'data': {},
    'model': {},
    'optimizer': {},
    'loss': {
        'local_lddt': {},
        'global_lddt': {},
        'global_gdtts': {},
    },
    'metric': {},
    'history': [],

    # Session defaults
    'session': {
        'max_epochs': 1,
        'batch_size': 1,
        'seed': random.randint(0, 99),
        'cpus': multiprocessing.cpu_count() - 1,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'log': [],
        'checkpoint': -1,
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
validate_losses(ex['loss'])

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
session['metric'] = {}
session['checkpoint'] = session['checkpoint']

if session['cpus'] < 0:
    raise ValueError(f'Invalid number of cpus: {session["cpus"]}')
if session['seed'] is None:
    raise ValueError(f'Invalid seed: {session["seed"]}')

ex['history'].append(session)

# Print config so far
sort_dict(ex, ['name', 'tags', 'fullname', 'completed_epochs', 'samples', 'data', 'model',
               'optimizer', 'loss', 'metric', 'history'])
sort_dict(session, ['completed_epochs', 'samples', 'max_epochs', 'batch_size', 'seed', 'cpus', 'device', 'status',
                    'datetime_started', 'datetime_completed', 'log', 'checkpoint', 'metric', 'git', 'gpus'])

pyaml.pprint(ex, safe=True, sort_dicts=False, force_embed=True, width=200)
# endregion

# region Building phase
# Random seeds (set them after the random run id is generated)
set_seeds(session['seed'])

# Saver
saver = Saver(Path(os.environ.get('RUNS_FOLDER', './runs')).joinpath(ex['fullname']))
if ex['completed_epochs'] == 0:
    saver.save_experiment(ex, epoch=ex['completed_epochs'], samples=ex['samples'])


# Model and optimizer
def load_model(config: Mapping) -> torch.nn.Module:
    # Reserved because used internally to fetch the model function/class and possibly load the weights
    reserved_keys = {'fn', 'state_dict'}
    # These args are computed based on other stuff, the user should not provide a value
    computed_keys = {'enc_in_nodes', 'enc_in_edges'}

    if 'fn' not in config:
        raise ValueError('Model function not specified')

    function = import_(config['fn'])
    function_args = inspect.signature(function).parameters.keys()
    if not set.isdisjoint(reserved_keys, function_args):
        raise ValueError(f'Model function can not have any of {reserved_keys} in its arguments, '
                         f'signature is {", ".join(function_args)}')
    if not set.isdisjoint(set(config.keys()), computed_keys):
        raise ValueError(f'Config dict can not have any of {computed_keys} in its arguments, '
                         f'found {", ".join(set.intersection(set(config.keys()), computed_keys))}')

    kwargs = {k: v for k, v in config.items() if k not in reserved_keys}
    model = function(
        enc_in_nodes=features.Input.Node.LENGTH + ex['data']['encoding_size'],
        enc_in_edges=features.Input.Edge.LENGTH,
        **kwargs
    )

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

# Logger: log experiment configuration and model parameters
logger = SummaryWriter(saver.base_folder)
logger.add_text('Experiment',
                textwrap.indent(pyaml.dump(ex, safe=True, sort_dicts=False, force_embed=True), '    '),
                global_step=ex['samples'])

print('Parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))
if ex['samples'] == 0:
    logger.add_scalar(
        'misc/parameters', sum(p.numel() for p in model.parameters() if p.requires_grad), global_step=ex['samples'])


# Datasets and dataloaders
def get_dataloaders(ex, session):
    data_folder = Path(os.environ.get('DATA_FOLDER', './data'))

    transforms = [
        RemoveEdges(cutoff=ex['data']['cutoff']),
        RbfDistEdges(sigma=ex['data']['cutoff']),
        PositionalEncoding(
            encoding_size=ex['data']['encoding_size'], max_sequence_length=900, base=ex['data']['encoding_base'])
    ]

    dataset_train = ProteinFolder(data_folder / 'training', transforms=transforms)
    dataset_val = ProteinFolder(data_folder / 'validation', transforms=transforms)

    if 'QUICK_RUN' in os.environ:
        print('QUICK RUN: limiting the datasets to 5 batches')
        dataset_train = torch.utils.data.Subset(dataset_train, range(5 * session['batch_size']))
        dataset_val = torch.utils.data.Subset(dataset_val, range(5 * session['batch_size']))

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
# endregion


# region Training
def training_function(trainer, batch):
    protein_names, model_names, graphs, targets = batch
    graphs = graphs.to(session['device'])
    targets = targets.to(session['device'])
    results = model(graphs)

    loss_local_lddt = torch.tensor(0., device=session['device'])
    loss_global_lddt = torch.tensor(0., device=session['device'])
    loss_global_gdtts = torch.tensor(0., device=session['device'])

    if ex['loss']['local_lddt']['weight'] > 0:
        node_mask = torch.isfinite(targets.node_features[:, features.Output.Node.LOCAL_LDDT])
        assert torch.isfinite(targets.node_features[node_mask, features.Output.Node.LOCAL_LDDT]).all().item()
        loss_local_lddt = F.mse_loss(
            results.node_features[node_mask, features.Output.Node.LOCAL_LDDT],
            targets.node_features[node_mask, features.Output.Node.LOCAL_LDDT], reduction='none'
        )
        if ex['loss']['local_lddt']['balanced']:
            loss_local_lddt = loss_local_lddt * targets.node_features[node_mask, features.Output.Node.LOCAL_LDDT_WEIGHT]
        loss_local_lddt = loss_local_lddt.mean()

    if ex['loss']['global_lddt']['weight'] > 0:
        loss_global_lddt = F.mse_loss(
            results.global_features[:, features.Output.Global.GLOBAL_LDDT],
            targets.global_features[:, features.Output.Global.GLOBAL_LDDT], reduction='none'
        )
        if ex['loss']['global_lddt']['balanced']:
            loss_global_lddt = loss_global_lddt * targets.global_features[:, features.Output.Global.GLOBAL_LDDT_WEIGHT]
        loss_global_lddt = loss_global_lddt.mean()

    if ex['loss']['global_gdtts']['weight'] > 0:
        loss_global_gdtts = F.mse_loss(
            results.global_features[:, features.Output.Global.GLOBAL_GDTTS],
            targets.global_features[:, features.Output.Global.GLOBAL_GDTTS], reduction='none'
        )
        if ex['loss']['global_gdtts']['balanced']:
            loss_global_gdtts = loss_global_gdtts * targets.global_features[:, features.Output.Global.GLOBAL_GDTTS_WEIGHT]
        loss_global_gdtts = loss_global_gdtts.mean()

    assert torch.isfinite(loss_local_lddt).item()
    assert torch.isfinite(loss_global_lddt).item()
    assert torch.isfinite(loss_global_gdtts).item()

    loss_total = (
        ex['loss']['local_lddt']['weight'] * loss_local_lddt +
        ex['loss']['global_lddt']['weight'] * loss_global_lddt +
        ex['loss']['global_gdtts']['weight'] * loss_global_gdtts
    )

    optimizer.zero_grad()
    loss_total.backward()
    optimizer.step(closure=None)

    return {
        'num_samples': len(graphs),
        'protein_names': protein_names,
        'model_names': model_names,
        'targets': targets,
        'results': results,
        'loss': {
            'total': loss_total.item(),
            'local_lddt': loss_local_lddt.item(),
            'global_lddt': loss_global_lddt.item(),
            'global_gdtts': loss_global_gdtts.item(),
        },
    }


def validation_function(validator, batch):
    protein_names, model_names, graphs, targets = batch
    graphs = graphs.to(session['device'])
    targets = targets.to(session['device'])
    results = model(graphs)

    loss_local_lddt = torch.tensor(0.)
    loss_global_lddt = torch.tensor(0.)
    loss_global_gdtts = torch.tensor(0.)

    if ex['loss']['local_lddt']['weight'] > 0:
        node_mask = torch.isfinite(targets.node_features[:, features.Output.Node.LOCAL_LDDT])
        loss_local_lddt = F.mse_loss(
            results.node_features[node_mask, features.Output.Node.LOCAL_LDDT],
            targets.node_features[node_mask, features.Output.Node.LOCAL_LDDT], reduction='none'
        )
        if ex['loss']['local_lddt']['balanced']:
            loss_local_lddt = loss_local_lddt * targets.node_features[node_mask, features.Output.Node.LOCAL_LDDT_WEIGHT]
        loss_local_lddt = loss_local_lddt.mean()

    if ex['loss']['global_lddt']['weight'] > 0:
        loss_global_lddt = F.mse_loss(
            results.global_features[:, features.Output.Global.GLOBAL_LDDT],
            targets.global_features[:, features.Output.Global.GLOBAL_LDDT], reduction='none'
        )
        if ex['loss']['global_lddt']['balanced']:
            loss_global_lddt = loss_global_lddt * targets.global_features[:, features.Output.Global.GLOBAL_LDDT_WEIGHT]
        loss_global_lddt = loss_global_lddt.mean()

    if ex['loss']['global_gdtts']['weight'] > 0:
        loss_global_gdtts = F.mse_loss(
            results.global_features[:, features.Output.Global.GLOBAL_GDTTS],
            targets.global_features[:, features.Output.Global.GLOBAL_GDTTS], reduction='none'
        )
        if ex['loss']['global_gdtts']['balanced']:
            loss_global_gdtts = loss_global_gdtts * targets.global_features[:, features.Output.Global.GLOBAL_GDTTS_WEIGHT]
        loss_global_gdtts = loss_global_gdtts.mean()

    loss_total = (
        ex['loss']['local_lddt']['weight'] * loss_local_lddt +
        ex['loss']['global_lddt']['weight'] * loss_global_lddt +
        ex['loss']['global_gdtts']['weight'] * loss_global_gdtts
    )

    return {
        'num_samples': len(graphs),
        'protein_names': protein_names, 
        'model_names': model_names,
        'targets': targets,
        'results': results,
        'loss': {
            'total': loss_total.item(),
            'local_lddt': loss_local_lddt.item(),
            'global_lddt': loss_global_lddt.item(),
            'global_gdtts': loss_global_gdtts.item(),
        },
    }


trainer = Engine(training_function)
validator = Engine(validation_function)


def session_start(trainer, session):
    session['status'] = 'RUNNING'
    session['datetime_started'] = datetime.utcnow()

    session_start_summary = make_session_start_summary(hparam_values={
        **{f'data/{k}': v for k, v in ex['data'].items()},
        **{f'optimizer/{k}': v for k, v in ex['optimizer'].items()},
        **{f'model/{k}': v for k, v in ex['model'].items()},
        **{f'loss/local_lddt/{k}': v for k, v in ex['loss']['local_lddt'].items()},
        **{f'loss/global_lddt/{k}': v for k, v in ex['loss']['global_lddt'].items()},
        **{f'loss/global_gdtts/{k}': v for k, v in ex['loss']['global_gdtts'].items()},
    })
    logger.file_writer.add_summary(session_start_summary)


def setup_training(trainer):
    model.train()
    torch.set_grad_enabled(True)


def update_samples(trainer: Engine, ex, session):
    num_samples = trainer.state.output['num_samples']
    ex['samples'] += num_samples
    session['samples'] += num_samples


def update_completed_epochs(trainer, ex, session):
    ex['completed_epochs'] += 1
    session['completed_epochs'] += 1


def save_model(trainer, model, ex, optimizer, session):
    if session['checkpoint'] != 0 and session['completed_epochs'] % session['checkpoint'] == 0:
        saver.save(model, ex, optimizer, epoch=ex['completed_epochs'], samples=ex['samples'])


def handle_failure(engine, e, name, ex, session):
    print(f'Exception raised during {name}, completed epochs {ex["completed_epochs"]}, samples {session["samples"]}')
    print(e)

    # Log session failure to tensorboard and to yaml
    session['status'] = 'FAILED'
    saver.save_experiment(ex, epoch=ex['completed_epochs'], samples=ex['samples'])

    logger.add_text('Experiment',
                    textwrap.indent(pyaml.dump(ex, safe=True, sort_dicts=False, force_embed=True), '    '),
                    ex['samples'])

    session_end_summary = make_session_end_summary('FAILURE')
    logger.file_writer.add_summary(session_end_summary)
    logger.close()

    raise e


losses_avg = ProteinAverageLosses(
    lambda o: (o['loss']['local_lddt'], o['loss']['global_lddt'], o['loss']['global_gdtts'], o['num_samples']))
losses_avg.attach(trainer)
losses_avg.attach(validator)

metrics = ProteinMetrics(itemgetter('protein_names', 'model_names', 'results', 'targets'))
metrics.attach(trainer)
metrics.attach(validator)

gpu_max_memory_allocated = GpuMaxMemoryAllocated()
gpu_max_memory_allocated.attach(trainer, 'misc/gpu')
gpu_max_memory_allocated.attach(validator, 'misc/gpu')

# During training and validation, the progress bar shows the average loss over all batches processed so far
pbar_train = ProgressBar(desc='Train')
pbar_train.attach(trainer, metric_names=losses_avg.losses_names)
pbar_val = ProgressBar(desc='Val')
pbar_val.attach(validator, metric_names=losses_avg.losses_names)


def log_losses_batch(engine, tag):
    """Log the losses for the current batch, to be called at every iteration"""
    for name, value in engine.state.output['loss'].items():
        if name == 'total' or ex['loss'][name]['weight'] > 0:
            logger.add_scalar(f'{tag}/loss/{name}', engine.state.output['loss'][name], global_step=session['samples'])


def log_losses_avg(engine, tag):
    """Log the avg of the losses for the current epoch, to be called at the end of an epoch"""
    for name, value in engine.state.metrics.items():
        if name.startswith('loss/') and ex['loss'][name[5:]]['weight'] > 0:
            logger.add_scalar(f'{tag}/{name}', value, global_step=session['samples'])


def log_metrics(engine, tag):
    for name, value in engine.state.metrics.items():
        if name.startswith('metric/'):
            logger.add_scalar(f'{tag}/{name}', value, global_step=session['samples'])


def log_misc(engine, tag):
    for name, value in engine.state.metrics.items():
        if name.startswith('misc/'):
            logger.add_scalar(f'{name}/{tag}', value, global_step=session['samples'])


def log_figures(engine, tag):
    for name, fig in engine.state.metrics.items():
        if name.startswith('fig/'):
            logger.add_figure(f'{tag}/{name[4:]}', fig, global_step=session['samples'], close=True)


def flush_logger(engine, logger):
    for writer in logger.all_writers.values():
        writer.flush()


def run_validation(trainer, validator, dataloader_val):
    validator.run(dataloader_val)


def session_end(trainer, session):
    session['status'] = 'COMPLETED'
    session['datetime_completed'] = datetime.utcnow()
    elapsed = session["datetime_completed"] - session["datetime_started"]

    print(f'Completed {session["completed_epochs"]} epochs in {round_timedelta(elapsed)} '
          f'({round_timedelta(elapsed / session["completed_epochs"])} per epoch)')

    if 'cuda' in session['device']:
        for device_id, device_info in session['cuda']['devices'].items():
            device_info.update({
                'memory_used_max': f'{torch.cuda.max_memory_allocated(device_id) // (10**6)} MiB',
                'memory_cached_max': f'{torch.cuda.max_memory_cached(device_id) // (10**6)} MiB',
            })
        print(pyaml.dump(session['cuda']['devices'], safe=True, sort_dicts=False), sep='\n')

    print(pyaml.dump(session['metric']))

    # Need to save again because we updated session and gpu info
    saver.save_experiment(ex, epoch=ex['completed_epochs'], samples=ex['samples'])

    # Log session end to tensorboard
    logger.add_text('Experiment',
                    textwrap.indent(pyaml.dump(ex, safe=True, sort_dicts=False, force_embed=True), '    '),
                    ex['samples'])
    session_end_summary = make_session_end_summary('SUCCESS')
    logger.file_writer.add_summary(session_end_summary)


trainer.add_event_handler(Events.STARTED, session_start, session)
trainer.add_event_handler(Events.EPOCH_STARTED, setup_training)
trainer.add_event_handler(Events.EXCEPTION_RAISED, handle_failure, 'training', ex, session)

trainer.add_event_handler(Events.ITERATION_COMPLETED, update_samples, ex, session)
trainer.add_event_handler(Events.ITERATION_COMPLETED, log_losses_batch, 'train')

trainer.add_event_handler(Events.EPOCH_COMPLETED, update_completed_epochs, ex, session)
trainer.add_event_handler(Events.EPOCH_COMPLETED, log_metrics, 'train')
trainer.add_event_handler(Events.EPOCH_COMPLETED, log_figures, 'train')
trainer.add_event_handler(Events.EPOCH_COMPLETED, log_misc, 'train')
trainer.add_event_handler(Events.EPOCH_COMPLETED, flush_logger, logger)
trainer.add_event_handler(Events.EPOCH_COMPLETED, run_validation, validator, dataloader_val)

trainer.add_event_handler(Events.COMPLETED, session_end, session)


def setup_validation(validator):
    model.eval()
    torch.set_grad_enabled(False)


def update_metrics(validator, ex, session):
    metrics = build_dict((k.split('/'), v) for k, v in validator.state.metrics.items() if k.startswith('metric/'))
    ex['metric'] = session['metric'] = metrics['metric']


validator.add_event_handler(Events.EPOCH_STARTED, setup_validation)
trainer.add_event_handler(Events.EXCEPTION_RAISED, handle_failure, 'validation', ex, session)

validator.add_event_handler(Events.EPOCH_COMPLETED, log_losses_avg, 'val')
validator.add_event_handler(Events.EPOCH_COMPLETED, log_metrics, 'val')
validator.add_event_handler(Events.EPOCH_COMPLETED, log_figures, 'val')
validator.add_event_handler(Events.EPOCH_COMPLETED, log_misc, 'val')
validator.add_event_handler(Events.EPOCH_COMPLETED, flush_logger, logger)
validator.add_event_handler(Events.EPOCH_COMPLETED, update_metrics, ex, session)
validator.add_event_handler(Events.EPOCH_COMPLETED, save_model, model, ex, optimizer, session)

trainer.run(dataloader_train, max_epochs=session['max_epochs'])
logger.close()
