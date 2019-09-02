import sys
import pyaml
import pathlib
import textwrap

from typing import Optional

import torch
import matplotlib.pyplot as plt

from torch.nn import Module
from ignite.engine import Engine
from tensorboardX import SummaryWriter

from .saver import Saver
from .config import build_dict
from .my_hparams import make_session_end_summary


def setup_training(trainer: Engine, model: Module):
    model.train()
    torch.set_grad_enabled(True)


def setup_validation(validator: Engine, model: Module):
    model.eval()
    torch.set_grad_enabled(False)


def setup_testing(tester: Engine, model: Module):
    model.eval()
    torch.set_grad_enabled(False)


def update_metrics(engine: Engine, session: dict):
    metrics = build_dict((k.split('/'), v) for k, v in engine.state.metrics.items())
    session['metric'] = metrics


def save_figures(engine: Engine, output_path: pathlib.Path):
    for name, fig in engine.state.figures.items():
        name = name.replace('/', '_')
        fig.savefig(output_path.joinpath(name).with_suffix('.pdf'))
        plt.close(fig)


def handle_failure(engine: Engine, e: Exception, name: str, ex: dict, session: dict,
                   saver: Optional[Saver] = None, logger: Optional[SummaryWriter] = None):
    """Log session failure to stderr, yaml and tensorboard"""
    session['status'] = 'FAILED'

    msg = f'Exception raised during {name}'
    if 'completed_epochs' in session:
        msg += f', session completed epochs {session["completed_epochs"]}'
    if 'samples' in session:
        msg += f', session samples {session["samples"]}'
    print(msg, file=sys.stderr)

    if saver is not None:
        saver.save_experiment(ex, epoch=ex['completed_epochs'], samples=ex['samples'])

    if logger is not None:
        logger.add_text('Experiment',
                        textwrap.indent(pyaml.dump(ex, safe=True, sort_dicts=False, force_embed=True), '    '),
                        ex['samples'])
        session_end_summary = make_session_end_summary('FAILURE')
        logger.file_writer.add_summary(session_end_summary)
        logger.flush()

    raise e


def flush_logger(engine: Engine, logger: SummaryWriter):
    logger.flush()
