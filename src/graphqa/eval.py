from argparse import ArgumentParser
from pathlib import Path
from typing import Union

import torch
from torch_geometric.data import DataLoader

from graphqa.data.paths import dataset_paths
from graphqa.utils.casp import scores_df_to_casp
from .dataset import DecoyDataset
from .train import LightningGraphQA


def load_model(checkpoint_path: Union[str, Path]):
    """Load model for evaluation from checkpoint."""
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    model = LightningGraphQA.load_from_checkpoint(checkpoint_path)
    model.eval()
    model.freeze()
    return model


def main():
    parser = ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("data_dir")
    parser.add_argument("--output_dir", required=False)
    parser.add_argument("--stage", type=int, default=2)
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    model = load_model(args.checkpoint)

    paths = dataset_paths(args.data_dir)
    ds = DecoyDataset(list(paths.processed.glob("*.pth")))
    dl = DataLoader(
        ds,
        drop_last=False,
        batch_size=model.conf.dataloader.batch_size,
        num_workers=0,#model.conf.dataloader.num_workers,
        pin_memory="cuda" in model.conf.session.device,
    )

    outputs = []
    for idx, batch in enumerate(dl):
        out = model.eval_step(batch, idx)
        outputs.append(out)
    scores = model.eval_epoch_end(outputs)
    scores_global = scores["global"].droplevel(level=1, axis=1)
    scores_local = scores["local"].droplevel(level=1, axis=1)

    if args.output_dir is None:
        args.output_dir = paths.root / 'predictions'
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    scores_global.to_csv(output_dir / "global.csv")
    scores_local.to_csv(output_dir / "local.csv")

    for target_id, target_str in scores_df_to_casp(
        args.stage, scores_global, scores_local
    ):
        qa_path = output_dir.joinpath(f"{target_id}.stage{args.stage}.qa")
        with qa_path.open("w") as f:
            f.write(target_str)


"""
python -m grapqa.eval 'runs/wandb/quirky_stallman_1133/checkpoints/graphqa_epoch=599.ckpt' 'data/COVID19-2' '/tmp/covid'
"""
if __name__ == "__main__":
    main()
