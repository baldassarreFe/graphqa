import itertools
import random
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import wandb
from loguru import logger
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch_geometric.data import Batch
from torch_geometric.data import DataLoader

from .dataset import DecoyDataset, RandomTargetSampler
from .logging import setup_logging, add_logfile
from .losses import nan_mse
from .metrics import scores_from_outputs, metrics_from_scores, figures_from_scores, log_dict_from_metrics
from .networks import GraphQA
from .utils.config import parse_config
from .utils.logging import figure_to_image


class LightningGraphQA(pl.LightningModule):
    def __init__(self, hparams: OmegaConf):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.conf = self.hparams

        self.model = torch.jit.script(GraphQA(self.conf.model))
        self.pth_files: Dict[str, Sequence[Path]] = {}

    def configure_optimizers(self):
        optimizer = Adam(
            self.parameters(),
            lr=self.conf.optimizer.lr,
            weight_decay=self.conf.optimizer.weight_decay,
        )
        scheduler = StepLR(
            optimizer,
            step_size=self.conf.scheduler.step_size,
            gamma=self.conf.scheduler.gamma,
        )
        return [optimizer], [scheduler]

    def prepare_data(self):
        data_root = Path(self.conf.datasets.folder)

        def glob_pth(casp_editions):
            return list(
                itertools.chain(
                    *(
                        data_root.glob(f"CASP{casp_ed}/processed/*.pth")
                        for casp_ed in casp_editions
                    )
                )
            )

        if "trainval" in self.conf.datasets:
            rg = np.random.RandomState(self.conf.datasets.trainval.seed)
            pth_paths = glob_pth(self.conf.datasets.trainval.casp_editions)
            pth_train, pth_val = train_test_split(
                pth_paths, train_size=self.conf.datasets.trainval.split, random_state=rg
            )
            self.pth_files["train"] = pth_train
            self.pth_files["val"] = pth_val

        if "train" in self.conf.datasets:
            self.pth_files["train"] = glob_pth(self.conf.datasets.train.casp_editions)

        if "val" in self.conf.datasets:
            self.pth_files["val"] = glob_pth(self.conf.datasets.val.casp_editions)

        if "test" in self.conf.datasets:
            self.pth_files["test"] = glob_pth(self.conf.datasets.test.casp_editions)

        for k, v in self.pth_files.items():
            logger.info(f"Pth files for {k}: {len(v)}")

    def train_dataloader(self) -> DataLoader:
        ds = DecoyDataset(self.pth_files["train"])
        logger.info(f"Training dataset: {len(ds)} decoys")
        rg = np.random.default_rng(self.conf.session.seed)
        dl = DataLoader(
            ds,
            drop_last=True,
            sampler=RandomTargetSampler(ds, rg),
            batch_size=self.conf.dataloader.batch_size,
            num_workers=self.conf.dataloader.num_workers,
            pin_memory="cuda" in self.conf.session.device,
        )
        return dl

    def val_dataloader(self) -> DataLoader:
        ds = DecoyDataset(self.pth_files["val"])
        logger.info(f"Validation dataset: {len(ds)} decoys")
        dl = DataLoader(
            ds,
            drop_last=False,
            batch_size=self.conf.dataloader.batch_size,
            num_workers=self.conf.dataloader.num_workers,
            pin_memory="cuda" in self.conf.session.device,
        )
        return dl

    def test_dataloader(self) -> DataLoader:
        ds = DecoyDataset(self.pth_files["test"])
        logger.info(f"Test dataset: {len(ds)} decoys")
        dl = DataLoader(
            ds,
            drop_last=False,
            batch_size=self.conf.dataloader.batch_size,
            num_workers=self.conf.dataloader.num_workers,
            pin_memory="cuda" in self.conf.session.device,
        )
        return dl

    def on_train_start(self) -> None:
        logger.info(
            f"Train start: epoch {self.trainer.current_epoch}, "
            f"step {self.trainer.global_step}"
        )

    def on_train_epoch_start(self) -> None:
        logger.info(
            f"Epoch start: epoch {self.trainer.current_epoch}, step {self.trainer.global_step}"
        )

        # Log learning rate
        self.log_dict(
            {
                f"optim/lr_{i}_{j}": pg["lr"]
                for i, optimizer in enumerate(self.trainer.optimizers)
                for j, pg in enumerate(optimizer.param_groups)
            },
        )

    def training_step(self, graphs: Batch, batch_idx: int):
        x, u = self(graphs)

        mse_local = nan_mse(x, graphs.qa_local)
        mse_global = nan_mse(u, graphs.qa_global)
        loss = (
            self.conf.losses.weight_local * mse_local.sum()
            + self.conf.losses.weight_global * mse_global.sum()
        )

        # When doing grad accumulation every 4 steps, the trainer will continue processing batches until
        # the dataloader is exhausted (e.g. 14 batches). The first 3*4 batches are actually used to step
        # the optimizer, the remaining 2 are just a waste of computation. Importantly, metrics from those
        # batches should not be logged or wandb will complain.
        #
        # An easy choice is to only log train metrics for the first batch of the accumulation batches.
        # In the past, the average value of each metric over all accumulation batches was computed,
        # but that required too much bookkeeping.
        if batch_idx % self.trainer.accumulate_grad_batches != 0:
            train_metrics = {
                "train/loss": loss.item(),
                "train/local/lddt": mse_local[0].item(),
                "train/local/cad": mse_local[1].item(),
                "train/global/tmscore": mse_global[0].item(),
                "train/global/gdtts": mse_global[1].item(),
                "train/global/gdtha": mse_global[2].item(),
                "train/global/lddt": mse_global[3].item(),
                "train/global/cad": mse_global[4].item(),
            }
            self.log_dict(train_metrics)

        return {"loss": loss}

    def training_epoch_end(self, outputs) -> None:
        logger.info(
            f"Epoch end: epoch {self.trainer.current_epoch}, "
            f"step {self.trainer.global_step}"
        )

    def on_train_end(self) -> None:
        logger.info(
            f"Train end: epoch {self.trainer.current_epoch}, "
            f"step {self.trainer.global_step}"
        )

    def on_validation_start(self) -> None:
        logger.info(
            f"Val start: epoch {self.trainer.current_epoch}, "
            f"step {self.trainer.global_step}"
        )

    def validation_step(self, graphs: Batch, batch_idx: int):
        x, u = self(graphs)

        scores_global = {
            "target_id": graphs.target_id,
            "decoy_id": graphs.decoy_id,
            ("tm", "true"): graphs.qa_global[:, 0].cpu().numpy(),
            ("tm", "pred"): u[:, 0].cpu().numpy(),
            ("gdtts", "true"): graphs.qa_global[:, 1].cpu().numpy(),
            ("gdtts", "pred"): u[:, 1].cpu().numpy(),
            ("gdtha", "true"): graphs.qa_global[:, 2].cpu().numpy(),
            ("gdtha", "pred"): u[:, 2].cpu().numpy(),
            ("lddt", "true"): graphs.qa_global[:, 3].cpu().numpy(),
            ("lddt", "pred"): u[:, 3].cpu().numpy(),
            ("cad", "true"): graphs.qa_global[:, 4].cpu().numpy(),
            ("cad", "pred"): u[:, 4].cpu().numpy(),
        }

        scores_local = {
            "target_id": np.array(graphs.target_id).repeat(graphs.n_nodes.cpu()),
            "decoy_id": np.array(graphs.decoy_id).repeat(graphs.n_nodes.cpu()),
            ("lddt", "true"): graphs.qa_local[:, 0].cpu().numpy(),
            ("lddt", "pred"): x[:, 0].cpu().numpy(),
            ("cad", "true"): graphs.qa_local[:, 1].cpu().numpy(),
            ("cad", "pred"): x[:, 1].cpu().numpy(),
        }

        return {"scores_global": scores_global, "scores_local": scores_local}

    def validation_epoch_end(self, outputs):
        logger.info(
            f"Val end: epoch {self.trainer.current_epoch}, "
            f"step {self.trainer.global_step}"
        )

        val_scores = scores_from_outputs(outputs)
        val_metrics = metrics_from_scores(val_scores)
        val_figures = figures_from_scores(val_scores)

        self.logger.experiment.log(
            {
                f"val/{key}": figure_to_image(fig, close=True)
                for key, fig in val_figures.items()
            },
        )

        val_metrics = log_dict_from_metrics(val_metrics, prefix="val")
        self.log_dict(val_metrics)

    def on_test_start(self) -> None:
        logger.info("Test start")

    def test_step(self, graphs: Batch, batch_idx: int):
        return self.validation_step(graphs, batch_idx)

    def test_epoch_end(self, outputs):
        logger.info("Test end")

        test_scores = scores_from_outputs(outputs)
        test_metrics = metrics_from_scores(test_scores)
        test_figures = figures_from_scores(test_scores)

        wandb.log(
            {
                f"test/{key}": figure_to_image(fig, close=False)
                for key, fig in test_figures.items()
            },
            step=self.trainer.global_step,
        )
        for key, fig in test_figures.items():
            fig.savefig(
                Path(wandb.run.dir).joinpath(key.replace("/", "_")).with_suffix(".png"),
                bbox_inches="tight",
                pad_inches=0.01,
                dpi=300,
            )
            plt.close(fig)

        test_metrics = log_dict_from_metrics(test_metrics, prefix="test")
        self.log_dict(test_metrics)
        # with self.run_dir.joinpath("metrics.csv").open("a") as f:
        #     f.write(f"epoch, {trainer.current_epoch}\n")
        #     f.write(f"step, {trainer.global_step}\n")
        #     for k, v in test_metrics.items():
        #         f.write(f"{k}, {v}\n")

        pd.to_pickle(
            test_scores,
            Path(wandb.run.dir).joinpath("test_scores.pkl"),
            compression="xz",
        )
        pd.to_pickle(
            test_metrics,
            Path(wandb.run.dir).joinpath("test_metrics.pkl"),
            compression="xz",
        )
        wandb.save("test_*.pkl")

        return test_metrics

    def eval_step(self, graphs: Batch, batch_idx: int):
        x, u = self(graphs)

        scores_global = {
            "target_id": graphs.target_id,
            "decoy_id": graphs.decoy_id,
            ("tm", "pred"): u[:, 0].cpu().numpy(),
            ("gdtts", "pred"): u[:, 1].cpu().numpy(),
            ("gdtha", "pred"): u[:, 2].cpu().numpy(),
            ("lddt", "pred"): u[:, 3].cpu().numpy(),
            ("cad", "pred"): u[:, 4].cpu().numpy(),
        }

        scores_local = {
            "target_id": np.array(graphs.target_id).repeat(graphs.n_nodes.cpu()),
            "decoy_id": np.array(graphs.decoy_id).repeat(graphs.n_nodes.cpu()),
            "residue_idx": np.concatenate([np.arange(n) for n in graphs.n_nodes.cpu()]),
            ("lddt", "pred"): x[:, 0].cpu().numpy(),
            ("cad", "pred"): x[:, 1].cpu().numpy(),
        }

        return {"scores_global": scores_global, "scores_local": scores_local}

    def eval_epoch_end(self, outputs):
        eval_scores = scores_from_outputs(outputs)
        return eval_scores

    def forward(self, graphs: Batch):
        inputs = GraphQA.prepare(graphs)
        outputs = self.model(*inputs)
        return outputs


def setup_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@logger.catch
def main():
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("--resume", required=False)
    parser.add_argument("rest", nargs="*")
    args = parser.parse_args()
    conf = parse_config(args.rest)
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(conf)}")

    setup_seeds(conf.session.seed)
    model = LightningGraphQA(conf)

    wandb_logger = WandbLogger(
        name=conf.fullname,
        save_dir=conf.checkpoint.folder,
        project=conf.project,
        tags=conf.tags,
        notes=conf.notes,
        config=OmegaConf.to_container(conf, resolve=True),
        config_exclude_keys=["notes", "tags", "project", "fullname"],
    )
    if rank_zero_only.rank == 0:
        add_logfile(Path(wandb_logger.experiment.dir) / "stdout.log")
        with open(Path(wandb_logger.experiment.dir) / "config.yaml", "w") as f:
            f.write(OmegaConf.to_yaml(conf))
        wandb.save("checkpoints/*")
        wandb.save("stdout.log*")
        wandb.save("config.yaml")

    model_dir = (
        (Path(wandb_logger.experiment.dir).parent / "checkpoints")
        if rank_zero_only.rank == 0
        else "/tmp"
    )
    checkpointer = ModelCheckpoint(
        dirpath=model_dir,
        save_top_k=conf.checkpoint.keep,
        verbose=True,
        monitor=conf.checkpoint.monitor,
        mode=conf.checkpoint.mode,
        prefix="graphqa",
        period=conf.checkpoint.period,
    )

    early_stop_callback = EarlyStopping(
        monitor=conf.session.early_stopping.monitor,
        min_delta=0.00,
        patience=conf.session.early_stopping.patience,
        verbose=True,
        mode=conf.session.early_stopping.mode,
    )

    trainer = pl.Trainer(
        gpus=1,
        max_epochs=conf.session.max_epochs,
        accumulate_grad_batches=conf.session.accumulate_grad,
        fast_dev_run=args.fast_dev_run,
        weights_summary=None,
        check_val_every_n_epoch=conf.session.val_every,
        logger=wandb_logger,
        callbacks=[early_stop_callback, checkpointer],
    )
    result = trainer.fit(model)
    if result == 1:
        trainer.test(model)


if __name__ == "__main__":
    setup_logging()
    main()
