from omegaconf import OmegaConf
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from proteins.config import flatten_dict
from proteins.hyperparameters import (
    add_hparam_summary,
    add_session_start,
    add_session_end,
)

METRICS = [
    {"tag": "val/global/gdtts/R", "display_name": "GDT-TS R"},
    {"tag": "val/global_per_target/gdtts/R", "display_name": "GDT-TS R target"},
    {"tag": "val/local/lddt/R", "display_name": "LDDT R"},
    {"tag": "val/local_per_decoy/lddt/R", "display_name": "LDDT R decoy"},
]


class MyTensorBoardLogger(TensorBoardLogger):
    def log_hyperparams(self, conf: OmegaConf) -> None:
        hparams = {
            "/".join(k): str(v) if isinstance(v, (list, tuple)) else v
            for k, v in flatten_dict(OmegaConf.to_container(conf, resolve=True))
        }

        writer = self.experiment
        add_hparam_summary(writer, hparams, METRICS)
        add_session_start(writer, hparams)

    def finalize(self, status: str) -> None:
        writer = self.experiment
        add_session_end(writer, status)
        self.save()


class MyEarlyStopping(EarlyStopping):
    def on_epoch_end(self, trainer, pl_module):
        pass

    def on_validation_end(self, trainer, pl_module):
        super(MyEarlyStopping, self).on_epoch_end(trainer, pl_module)
