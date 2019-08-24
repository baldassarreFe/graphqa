import os
import yaml
import pyaml
import multiprocessing

from pathlib import Path
from datetime import datetime
from operator import itemgetter

import numpy as np
import pandas as pd

import torch
import torch.utils.data
import torchgraphs as tg

from ignite.engine import Engine, Events
from ignite.contrib.handlers import ProgressBar

from .config import build_dict
from .config import parse_args
from .utils import git_info, cuda_info, sort_dict, round_timedelta, load_model
from .dataset import ProteinQualityDataset, PositionalEncoding, RemoveEdges, RbfDistEdges, SeparationEncoding
from .metrics import ProteinMetrics

# region Arguments parsing
ex = parse_args(config={
    # Experiment defaults
    'model': {},
    'data': {},

    # Test session defaults
    'test': {
        'data': {
            'input': os.environ.get('DATA_FOLDER', './data'),
            'output': os.environ.get('OUTPUT_FOLDER', './test')
        },
        'batch_size': 1,
        'cpus': multiprocessing.cpu_count() - 1,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }
})
session: dict = ex['test']


# Experiment: checks and computed fields
if ex['model']['fn'] is None:
    raise ValueError('Model constructor function not defined')

# Session computed fields
session['samples'] = 0
session['status'] = 'NEW'
session['datetime_started'] = None
session['datetime_completed'] = None
session['git'] = git_info()
session['cuda'] = cuda_info() if 'cuda' in session['device'] else None
session['metric'] = {}

if session['cpus'] < 0:
    raise ValueError(f'Invalid number of cpus: {session["cpus"]}')

# Print config so far
sort_dict(ex, ['name', 'tags', 'fullname', 'completed_epochs', 'samples', 'data', 'model',
               'optimizer', 'loss', 'metric', 'history', 'test'])
sort_dict(session, ['data', 'batch_size', 'cpus', 'device', 'samples', 'status', 'datetime_started',
                    'datetime_completed', 'metric', 'git', 'cuda'])

pyaml.pprint(ex, safe=True, sort_dicts=False, force_embed=True, width=200)
# endregion

# region Building phase


# Saver
Path(session['data']['output']).expanduser().resolve().mkdir(parents=True, exist_ok=True)
with open(Path(session['data']['output']).expanduser().resolve() / 'test.yaml', 'w') as f:
    pyaml.dump(ex, f, safe=True, sort_dicts=False, force_embed=True)


# Model and optimizer
model = load_model(ex['model']).to(session['device'])
model.load_state_dict(torch.load(Path(ex['model']['state_dict']).expanduser(), map_location=session['device']))


# Dataset and dataloader
def get_dataloader(ex, session):
    folder = Path(session['data']['input']).expanduser().resolve()
    if not folder.is_dir():
        raise ValueError(f'Not a directory: {folder}')

    with open(folder / 'dataset_stats.yaml') as f:
        max_sequence_length = yaml.safe_load(f)['max_length']
    df = pd.read_csv(folder / 'samples.csv', header=0)
    df['path'] = [folder / p for p in df['path']]

    transforms = [
        RemoveEdges(cutoff=ex['data']['cutoff']),
        RbfDistEdges(sigma=ex['data']['sigma']),
        SeparationEncoding(use_separation=ex['data']['separation']),
        PositionalEncoding(encoding_size=ex['data']['encoding_size'], base=ex['data']['encoding_base'],
                           max_sequence_length=max_sequence_length)
    ]

    dataset_test = ProteinQualityDataset(df, transforms=transforms)

    dataloader_kwargs = dict(
        num_workers=session['cpus'],
        pin_memory='cuda' in session['device'],
        worker_init_fn=lambda _: np.random.seed(int(torch.initial_seed()) % (2 ** 32 - 1)),
        batch_size=session['batch_size'],
        collate_fn=tg.GraphBatch.collate,
    )
    return torch.utils.data.DataLoader(dataset_test, shuffle=False, **dataloader_kwargs)


dataloader_test = get_dataloader(ex, session)
# endregion


def test_function(tester, batch):
    protein_names, model_names, graphs, targets = batch
    graphs = graphs.to(session['device'])
    targets = targets.to(session['device'])
    results = model(graphs)

    return {
        'num_samples': len(graphs),
        'protein_names': protein_names, 
        'model_names': model_names,
        'targets': targets,
        'results': results,
    }


def session_start(tester, session):
    session['status'] = 'RUNNING'
    session['datetime_started'] = datetime.utcnow()

    with open(Path(session['data']['output']).expanduser().resolve() / 'test.yaml', 'w') as f:
        pyaml.dump(ex, f, safe=True, sort_dicts=False, force_embed=True)


def setup_testing(tester):
    model.eval()
    torch.set_grad_enabled(False)


def update_samples(tester: Engine, ex, session):
    session['samples'] += tester.state.output['num_samples']


def handle_failure(engine, e, name, ex, session):
    print(f'Exception raised during {name}, samples {session["samples"]}')
    print(e)

    # Log session failure to tensorboard and to yaml
    session['status'] = 'FAILED'
    with open(Path(session['data']['output']).expanduser().resolve() / 'test.yaml', 'w') as f:
        pyaml.dump(ex, f, safe=True, sort_dicts=False, force_embed=True)

    raise e

tester = Engine(test_function)

metrics = ProteinMetrics(itemgetter('protein_names', 'model_names', 'results', 'targets'))
metrics.attach(tester)

# During testing, the progress bar shows the number of batches processed so far
pbar_test = ProgressBar(desc='Test')
pbar_test.attach(tester)


def update_metrics(tester, ex, session):
    metrics = build_dict((k.split('/'), v) for k, v in tester.state.metrics.items() if k.startswith('metric/'))
    session['metric'] = metrics['metric']


def save_figures(engine):
    for name, fig in engine.state.metrics.items():
        if name.startswith('fig/'):
            fig.savefig(Path(session['data']['output']).expanduser().resolve() / f'{name[4:]}.png')


def session_end(tester, session):
    session['status'] = 'COMPLETED'
    session['datetime_completed'] = datetime.utcnow()
    elapsed = session["datetime_completed"] - session["datetime_started"]

    print(f'Tested {session["samples"]} protein models in {round_timedelta(elapsed)}')

    if 'cuda' in session['device']:
        for device_id, device_info in session['cuda']['devices'].items():
            device_info.update({
                'memory_used_max': f'{torch.cuda.max_memory_allocated(device_id) // (10**6)} MiB',
                'memory_cached_max': f'{torch.cuda.max_memory_cached(device_id) // (10**6)} MiB',
            })
        print(pyaml.dump(session['cuda']['devices'], safe=True, sort_dicts=False), sep='\n')

    print(pyaml.dump(session['metric']))

    # Need to save again because we updated session and gpu info
    with open(Path(session['data']['output']).expanduser().resolve() / 'test.yaml', 'w') as f:
        pyaml.dump(ex, f, safe=True, sort_dicts=False, force_embed=True)


tester.add_event_handler(Events.STARTED, session_start, session)
tester.add_event_handler(Events.STARTED, setup_testing)
tester.add_event_handler(Events.EXCEPTION_RAISED, handle_failure, 'testing', ex, session)

tester.add_event_handler(Events.ITERATION_COMPLETED, update_samples, ex, session)

tester.add_event_handler(Events.COMPLETED, update_metrics, ex, session)
tester.add_event_handler(Events.COMPLETED, save_figures)
tester.add_event_handler(Events.COMPLETED, session_end, session)

tester.run(dataloader_test)
