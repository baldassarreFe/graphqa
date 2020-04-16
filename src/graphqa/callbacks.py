from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import wandb
from loguru import logger
from omegaconf import OmegaConf
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from .hyperparameters import add_hparam_summary, add_session_start, add_session_end
from .metrics import log_dict_from_metrics
from .utils.config import flatten_dict


class MyModelCheckpoint(ModelCheckpoint):
    def on_validation_end(self, trainer, pl_module):
        callback_metrics_old = trainer.callback_metrics
        trainer.callback_metrics = log_dict_from_metrics(
            trainer.callback_metrics["val_metrics"], prefix="val"
        )
        super(MyModelCheckpoint, self).on_validation_end(trainer, pl_module)
        trainer.callback_metrics = callback_metrics_old


class MyEarlyStopping(EarlyStopping):
    def on_epoch_end(self, trainer, pl_module):
        pass

    def on_validation_end(self, trainer, pl_module):
        super(MyEarlyStopping, self).on_epoch_end(trainer, pl_module)

    def check_metrics(self, logs):
        logs = log_dict_from_metrics(logs["val_metrics"], prefix="val")
        return super(MyEarlyStopping, self).check_metrics(logs)


def wandb_init(conf: OmegaConf):
    wandb.init(
        name=conf.fullname,
        id=conf.fullname,
        dir=Path(conf.checkpoint.folder).expanduser().resolve().as_posix(),
        project=conf.project,
        tags=conf.tags,
        notes=conf.notes,
        config=OmegaConf.to_container(conf, resolve=True),
        config_exclude_keys=["notes", "tags", "project", "fullname"],
    )
    wandb.save("checkpoints/*", policy="live")
    wandb.save("stdout.log", policy="live")
    wandb.save("conf_*.yaml", policy="live")


class WandbCallbacks(pl.callbacks.Callback):
    metrics_train_batches: List[Dict[str, Any]]

    def __init__(self, conf: OmegaConf):
        self.metrics_train_batches = []

    @staticmethod
    def _figure_to_image(fig: plt.Figure, close: bool):
        from torch.utils.tensorboard._utils import figure_to_image

        img = figure_to_image(fig, close=close)
        img = np.moveaxis(img, 0, -1)
        img = wandb.Image(img)
        return img

    @property
    def log_dir(self):
        return Path(wandb.run.dir)

    def on_train_start(self, trainer: pl.Trainer, pl_module: LightningModule):
        logger.info(
            f"Train start: epoch {trainer.current_epoch}, step {trainer.global_step}"
        )
        # with self.run_dir.joinpath(f"conf_{trainer.global_step}.yaml").open("a") as f:
        #     f.write(pl_module.conf.pretty())

    def on_epoch_start(self, trainer: pl.Trainer, pl_module: LightningModule):
        logger.info(
            f"Epoch start: epoch {trainer.current_epoch}, step {trainer.global_step}"
        )

        # Reset training metrics
        self.metrics_train_batches.clear()

        # Log learning rate
        wandb.log(
            {
                f"optim/lr_{i}_{j}": pg["lr"]
                for i, optimizer in enumerate(trainer.optimizers)
                for j, pg in enumerate(optimizer.param_groups)
            },
            step=trainer.global_step,
        )

    def on_batch_end(self, trainer: pl.Trainer, pl_module: LightningModule):
        num_batches_in_epoch = (
            trainer.num_training_batches // trainer.accumulate_grad_batches
        ) * trainer.accumulate_grad_batches
        if trainer.batch_idx > num_batches_in_epoch:
            # When doing grad accumulation every 4 steps, the trainer will continue processing batches until
            # the dataloader is exhausted (e.g. 14 batches). The first 3*4 batches are actually used to step
            # the optimizer, the remaining 2 are just a waste of computation. Importantly, metrics from those
            # batches should not be logged or wandb will complain.
            return

        if trainer.accumulate_grad_batches > 1:
            self.metrics_train_batches.append(trainer.hiddens["train_metrics"])
            if len(self.metrics_train_batches) == trainer.accumulate_grad_batches:
                aggregation = {
                    k: np.mean([v[k] for v in self.metrics_train_batches])
                    for k in self.metrics_train_batches[0]
                }
                self.metrics_train_batches.clear()
                wandb.log(aggregation, step=trainer.global_step)
        else:
            wandb.log(trainer.hiddens["train_metrics"], step=trainer.global_step)

    def on_epoch_end(self, trainer: pl.Trainer, pl_module: LightningModule):
        logger.info(
            f"Epoch end: epoch {trainer.current_epoch}, step {trainer.global_step}"
        )
        pass

    def on_validation_start(self, trainer: pl.Trainer, pl_module: LightningModule):
        logger.info(
            f"Val start: epoch {trainer.current_epoch}, step {trainer.global_step}"
        )

    def on_validation_end(self, trainer: pl.Trainer, pl_module: LightningModule):
        logger.info(
            f"Val end: epoch {trainer.current_epoch}, step {trainer.global_step}"
        )

        val_figures = trainer.callback_metrics["val_figures"]
        wandb.log(
            {
                f"val/{key}": self._figure_to_image(fig, close=True)
                for key, fig in val_figures.items()
            },
            step=trainer.global_step,
        )

        val_metrics = log_dict_from_metrics(
            trainer.callback_metrics["val_metrics"], prefix="val"
        )
        wandb.log(val_metrics, step=trainer.global_step)
        # with self.run_dir.joinpath("metrics.csv").open("a") as f:
        #     f.write(f"epoch, {trainer.current_epoch}\n")
        #     f.write(f"step, {trainer.global_step}\n")
        #     for k, v in val_metrics.items():
        #         f.write(f"{k}, {v}\n")

    def on_train_end(self, trainer: pl.Trainer, pl_module: LightningModule):
        logger.info(
            f"Train end: epoch {trainer.current_epoch}, step {trainer.global_step}"
        )
        # Not sure why sometimes the last validation doesn't get logged
        # val_metrics = log_dict_from_metrics(
        #     trainer.callback_metrics["val_metrics"], prefix="val"
        # )
        # wandb.log(val_metrics, step=trainer.global_step)

    def on_test_start(self, trainer: pl.Trainer, pl_module: LightningModule):
        logger.info("Test start")

    def on_test_end(self, trainer: pl.Trainer, pl_module: LightningModule):
        logger.info("Test end")

        test_figures = trainer.callback_metrics["test_figures"]
        wandb.log(
            {
                f"test/{key}": self._figure_to_image(fig, close=False)
                for key, fig in test_figures.items()
            },
            step=trainer.global_step,
        )
        for key, fig in test_figures.items():
            fig.savefig(
                Path(wandb.run.dir).joinpath(key.replace("/", "_")).with_suffix(".png"),
                bbox_inches="tight",
                pad_inches=0.01,
                dpi=300,
            )
            plt.close(fig)

        test_metrics = log_dict_from_metrics(
            trainer.callback_metrics["test_metrics"], prefix="test"
        )
        wandb.log(test_metrics, step=trainer.global_step)
        # with self.run_dir.joinpath("metrics.csv").open("a") as f:
        #     f.write(f"epoch, {trainer.current_epoch}\n")
        #     f.write(f"step, {trainer.global_step}\n")
        #     for k, v in test_metrics.items():
        #         f.write(f"{k}, {v}\n")

        pd.to_pickle(
            trainer.callback_metrics["test_scores"],
            Path(wandb.run.dir).joinpath("test_scores.pkl"),
            compression="xz",
        )
        pd.to_pickle(
            trainer.callback_metrics["test_metrics"],
            Path(wandb.run.dir).joinpath("test_metrics.pkl"),
            compression="xz",
        )
        wandb.save("test_*.pkl", policy="live")


class MyTensorBoardLogger(TensorBoardLogger):
    METRICS = [
        {"tag": "val/global/gdtts/R", "display_name": "GDT-TS R"},
        {"tag": "val/global_per_target/gdtts/R", "display_name": "GDT-TS R target"},
        {"tag": "val/local/lddt/R", "display_name": "LDDT R"},
        {"tag": "val/local_per_decoy/lddt/R", "display_name": "LDDT R decoy"},
    ]

    def log_hyperparams(self, conf: OmegaConf) -> None:
        hparams = {
            "/".join(k): str(v) if isinstance(v, (list, tuple)) else v
            for k, v in flatten_dict(OmegaConf.to_container(conf, resolve=True))
        }

        writer = self.experiment
        add_hparam_summary(writer, hparams, self.METRICS)
        add_session_start(writer, hparams)

    def finalize(self, status: str) -> None:
        writer = self.experiment
        add_session_end(writer, status)
        self.save()
