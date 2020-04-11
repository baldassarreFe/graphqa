import itertools
import random
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from loguru import logger
from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch_geometric.data import Batch
from torch_geometric.data import DataLoader

from proteins.callbacks import MyTensorBoardLogger, MyEarlyStopping
from proteins.config import parse_config
from proteins.dataset import DecoyDataset, RandomTargetSampler
from proteins.logging import setup_logging, add_logfile
from proteins.metrics import (
    scores_from_outputs,
    metrics_from_scores,
    figures_from_scores,
    log_dict_from_metrics,
)
from proteins.networks import GraphQA


@torch.jit.script
def nan_mse(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    mask = torch.isfinite(targets)
    zero = targets.new_zeros(())
    num = torch.where(mask, targets - inputs, zero).pow(2).sum(dim=0)
    den = mask.sum(dim=0)
    return num / den


class LightningGraphQA(pl.LightningModule):
    def __init__(self, hparams: OmegaConf):
        super().__init__()

        # # Workaround lightning loading strategy
        # if isinstance(hparams, Namespace):
        #     self._hparams = hparams
        #     self.conf = namespace_to_omegaconf(hparams)
        # elif OmegaConf.is_config(hparams):
        #     self.conf = hparams
        #     self._hparams = omegaconf_to_namespace(hparams)
        # else:
        #     raise ValueError(f"Invalid configuration {hparams}")

        self.hparams = self.conf = hparams
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

    def training_step(self, graphs: Batch, batch_idx: int):
        x, u = self(graphs)

        mse_local = nan_mse(x, graphs.qa_local)
        mse_global = nan_mse(u, graphs.qa_global)
        loss = (
            self.conf.losses.weight_local * mse_local.sum()
            + self.conf.losses.weight_global * mse_global.sum()
        )

        return {
            "loss": loss,
            "log": {
                "train/loss": loss.item(),
                "train/local/lddt": mse_local[0].item(),
                "train/local/cad": mse_local[1].item(),
                "train/global/tmscore": mse_global[0].item(),
                "train/global/gdtts": mse_global[1].item(),
                "train/global/gdtha": mse_global[2].item(),
                "train/global/lddt": mse_global[3].item(),
                "train/global/cad": mse_global[4].item(),
            },
        }

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
        val_scores = scores_from_outputs(outputs)
        val_metrics = metrics_from_scores(val_scores)
        val_figures = figures_from_scores(val_scores)
        return {
            "log": log_dict_from_metrics(val_metrics, prefix="val"),
            "val_scores": val_scores,
            "val_metrics": val_metrics,
            "val_figures": val_figures,
        }

    def test_step(self, graphs: Batch, batch_idx: int):
        return self.validation_step(graphs, batch_idx)

    def test_epoch_end(self, outputs):
        test_scores = scores_from_outputs(outputs)
        test_metrics = metrics_from_scores(test_scores)
        test_figures = figures_from_scores(test_scores)
        return {
            "log": log_dict_from_metrics(test_metrics, prefix="test"),
            "test_scores": test_scores,
            "test_metrics": test_metrics,
            "test_figures": test_figures,
        }

    def forward(self, graphs: Batch):
        inputs = GraphQA.prepare(graphs)
        outputs = self.model(*inputs)
        return outputs


def setup_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class LogsCallback(pl.callbacks.Callback):
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir.expanduser().resolve()

    def on_train_start(self, trainer: pl.Trainer, pl_module: LightningGraphQA):
        logger.info(
            f"Train start: epoch {trainer.current_epoch}, step {trainer.global_step}"
        )
        with self.run_dir.joinpath(f"conf_{trainer.global_step}.yaml").open("a") as f:
            f.write(pl_module.conf.pretty())

    # def on_epoch_start(self, trainer: pl.Trainer, pl_module: LightningGraphQA):
    #     logger.info("Epoch start")

    def on_epoch_end(self, trainer: pl.Trainer, pl_module: LightningGraphQA):
        # logger.info("Epoch end")
        trainer.logger.log_metrics(
            {
                f"lr_{i}_{j}": pg["lr"]
                for i, optimizer in enumerate(trainer.optimizers)
                for j, pg in enumerate(optimizer.param_groups)
            },
            step=trainer.global_step,
        )

    def on_validation_start(self, trainer: pl.Trainer, pl_module: LightningGraphQA):
        logger.info(
            f"Val start: epoch {trainer.current_epoch}, step {trainer.global_step}"
        )

    def on_validation_end(self, trainer: pl.Trainer, pl_module: LightningGraphQA):
        logger.info(
            f"Val end: epoch {trainer.current_epoch}, step {trainer.global_step}"
        )

        val_figures = trainer.callback_metrics["val_figures"]
        for key, fig in val_figures.items():
            trainer.logger.experiment.add_figure(
                f"val/{key}", fig, trainer.global_step, close=True
            )

        val_metrics = log_dict_from_metrics(
            trainer.callback_metrics["val_metrics"], prefix="val"
        )
        with self.run_dir.joinpath("metrics.csv").open("a") as f:
            f.write(f"epoch, {trainer.current_epoch}\n")
            f.write(f"step, {trainer.global_step}\n")
            for k, v in val_metrics.items():
                f.write(f"{k}, {v}\n")

    def on_train_end(self, trainer: pl.Trainer, pl_module: LightningGraphQA):
        logger.info(
            f"Train end: epoch {trainer.current_epoch}, step {trainer.global_step}"
        )
        # Not sure why the last validation doesn't get logged
        val_metrics = log_dict_from_metrics(
            trainer.callback_metrics["val_metrics"], prefix="val"
        )
        trainer.logger.log_metrics(val_metrics, step=trainer.global_step)

    def on_test_start(self, trainer: pl.Trainer, pl_module: LightningGraphQA):
        logger.info("Test start")

    def on_test_end(self, trainer: pl.Trainer, pl_module: LightningGraphQA):
        logger.info("Test end")

        test_figures = trainer.callback_metrics["test_figures"]
        for key, fig in test_figures.items():
            trainer.logger.experiment.add_figure(
                f"test/{key}", fig, trainer.global_step, close=False
            )
            fig.savefig(
                self.run_dir.joinpath(key.replace("/", "_")).with_suffix(".png"),
                bbox_inches="tight",
                pad_inches=0.01,
                dpi=300,
            )
            plt.close(fig)

        test_metrics = log_dict_from_metrics(
            trainer.callback_metrics["test_metrics"], prefix="test"
        )
        trainer.logger.log_metrics(test_metrics, step=trainer.global_step)
        with self.run_dir.joinpath("metrics.csv").open("a") as f:
            f.write(f"epoch, {trainer.current_epoch}\n")
            f.write(f"step, {trainer.global_step}\n")
            for k, v in test_metrics.items():
                f.write(f"{k}, {v}\n")

        pd.to_pickle(
            trainer.callback_metrics["test_scores"],
            self.run_dir.joinpath("test_scores.pkl"),
        )
        pd.to_pickle(
            trainer.callback_metrics["test_metrics"],
            self.run_dir.joinpath("test_metrics.pkl"),
        )


@logger.catch
def main():
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("--resume", required=False)
    parser.add_argument("rest", nargs="*")
    args = parser.parse_args()
    conf = parse_config(args.rest)

    add_logfile(Path(conf.checkpoint.folder) / conf.fullname / "logs.txt")
    setup_seeds(conf.session.seed)
    logger.info(f"Configuration:\n{conf.pretty()}")

    model = LightningGraphQA(conf)
    checkpointer = pl.callbacks.ModelCheckpoint(
        filepath=Path(conf.checkpoint.folder)
        / conf.fullname
        / "checkpoints"
        / "{epoch}",
        save_top_k=conf.checkpoint.keep,
        verbose=True,
        monitor=conf.checkpoint.monitor,
        mode=conf.checkpoint.mode,
        prefix="graphqa_",
        period=conf.checkpoint.period,
    )
    tb_logger = MyTensorBoardLogger(
        save_dir=Path(conf.checkpoint.folder), name=None, version=conf.fullname
    )
    early_stop_callback = MyEarlyStopping(
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
        log_gpu_memory=True,
        weights_summary=None,
        check_val_every_n_epoch=conf.session.val_every,
        logger=tb_logger,
        checkpoint_callback=checkpointer,
        early_stop_callback=early_stop_callback,
        callbacks=[LogsCallback(Path(conf.checkpoint.folder) / conf.fullname)],
    )
    trainer.fit(model)
    trainer.test(model)


if __name__ == "__main__":
    setup_logging()
    main()
