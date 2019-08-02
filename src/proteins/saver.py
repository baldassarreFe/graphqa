import os
from pathlib import Path
from typing import Union

import pyaml
import torch


class Saver(object):
    def __init__(self, folder: Union[str, Path]):
        self.base_folder = Path(folder).expanduser().resolve()
        self.checkpoint_folder = self.base_folder / 'checkpoints'

    def save_model(self, model, epoch, samples, is_best=False):
        self.checkpoint_folder.mkdir(parents=True, exist_ok=True)

        if isinstance(model, torch.nn.DataParallel):
            model = model.module
        name = f'model.e{epoch}.s{samples}.pt'
        model_path = self.checkpoint_folder / name
        torch.save(model.state_dict(), model_path)

        latest_path = self.base_folder / 'model.latest.pt'
        if latest_path.exists():
            os.unlink(latest_path)
        os.link(model_path, latest_path)

        if is_best:
            best_path = self.base_folder / 'model.best.pt'
            if best_path.exists():
                os.unlink(latest_path)
            os.link(model_path, best_path)

        return model_path.as_posix()

    def save_optimizer(self, optimizer, epoch, samples):
        self.checkpoint_folder.mkdir(parents=True, exist_ok=True)

        name = f'optimizer.e{epoch}.s{samples}.pt'
        optimizer_path = self.checkpoint_folder / name
        torch.save(optimizer.state_dict(), optimizer_path)

        latest_path = self.base_folder / 'optimizer.latest.pt'
        if latest_path.exists():
            os.unlink(latest_path)
        os.link(optimizer_path, latest_path)

        return optimizer_path.as_posix()

    def save_experiment(self, experiment, epoch, samples):
        self.checkpoint_folder.mkdir(parents=True, exist_ok=True)

        name = f'experiment.e{epoch}.s{samples}.yaml'
        experiment_path = self.checkpoint_folder / name
        with open(experiment_path, 'w') as f:
            pyaml.dump(experiment, f, safe=True, sort_dicts=False, force_embed=True)

        latest_path = self.base_folder / 'experiment.latest.yaml'
        if latest_path.exists():
            os.unlink(latest_path)
        os.link(experiment_path, latest_path)

        return experiment_path.as_posix()

    def save(self, model, experiment, optimizer, epoch, samples, is_best=False):
        return {
            'model': self.save_model(model, epoch, samples, is_best),
            'optimizer': self.save_optimizer(optimizer, epoch, samples),
            'experiment': self.save_experiment(experiment, epoch, samples)
        }
