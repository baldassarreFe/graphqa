import time
import pyaml
import argparse
from pathlib import Path

from tensorboardX import SummaryWriter

from .my_hparams import make_experiment_summary

parser = argparse.ArgumentParser()
parser.add_argument('folder', help='The log folder to place the experiment configuration in')
args = parser.parse_args()

hparam_infos = {
    'data': {
        'cutoff': {'type': float},
        'encoding_size': {'type': int},
        'encoding_base': {'type': float},
    },
    'optimizer': {
        'lr': {'type': float},
        'weight_decay': {'type': float},
    },
    'model': {
        'layers': {'type': int},
        'mp_in_edges': {'type': int},
        'mp_in_nodes': {'type': int},
        'mp_in_globals': {'type': int},
        'mp_out_edges': {'type': int},
        'mp_out_nodes': {'type': int},
        'mp_out_globals': {'type': int},
        'dropout': {'type': float},
        'batch_norm': {'type': bool},
    },
    'loss/local_lddt': {
        'name': {'type': str},
        'weight': {'type': float},
        'balanced': {'type': bool},
    },
    'loss/global_lddt': {
        'name': {'type': str},
        'weight': {'type': float},
        'balanced': {'type': bool},
    },
    'loss/global_gdtts': {
        'name': {'type': str},
        'weight': {'type': float},
        'balanced': {'type': bool},
    }
}

metric_infos = {
    'local_lddt': {
        'r2',
        'rmse',
        'correlation',
        'correlation_per_model',
    },
    'global_lddt': {
        'r2',
        'rmse',
        'correlation',
        'correlation_per_target',
        'first_rank_loss',
    },
    'global_gdtts': {
        'r2',
        'rmse',
        'correlation',
        'correlation_per_target',
        'first_rank_loss',
    }
}

hparam_infos = [
    {'name': f'{category}/{hp_name}', **hp_info}
    for category, hparams in hparam_infos.items()
    for hp_name, hp_info in hparams.items()
]

metric_infos = [
    {
        'tag': f'val/metric/{score_type}/{n}',
        'display_name': f'{score_type}/{n}',
        'dataset_type': 'validation'
    }
    for score_type, names in metric_infos.items()
    for n in names
]

experiment = {
    'name': 'proteins',
    'time_created_secs': int(time.time())
}

pyaml.add_representer(type, lambda dumper, data: dumper.represent_str(data.__name__))
pyaml.print({'experiment': experiment, 'hparams': hparam_infos, 'metrics': metric_infos},
            safe=True, sort_dicts=False, force_embed=True)

folder = (Path(args.folder) / 'experiment').expanduser().resolve()
with SummaryWriter(folder) as writer:
    experiment_summary = make_experiment_summary(hparam_infos, metric_infos, experiment)
    writer.file_writer.add_summary(experiment_summary)

print('Experiment summary saved to', folder)
