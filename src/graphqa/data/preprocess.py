import argparse
import multiprocessing
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from loguru import logger

from graphqa.data.cadscore import run_cadscore
from graphqa.data.lddt import run_lddt
from graphqa.data.tmscore import run_tmscore
from .msa import run_msa, compute_msa_features
from .dssp import run_dssp
from .paths import dataset_paths
from .decoys import load_decoy_feats
from .graphs import ProteinGraphBuilder
from .sequences import FastaToNumpyWrapper


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", help="Data dir.")
    parser.add_argument("uniref", help="Uniref database.")

    parser.add_argument(
        "--train",
        action="store_true",
        help="Run additional steps to compute ground-truth scores",
    )
    parser.add_argument(
        "--tmscore", default="TMscore", help="TMscore executable if not in PATH"
    )
    parser.add_argument(
        "--voronota",
        default="voronota-cadscore",
        help="Voronota executable if not in PATH",
    )

    parser.add_argument(
        "--cpus",
        default=max(1, multiprocessing.cpu_count() - 1),
        type=int,
        help="CPUs to use",
    )
    # TODO add argument --skip-existing

    args = parser.parse_args()
    paths = dataset_paths(args.dataset_dir)
    if args.train and not paths.native.is_dir():
        raise ValueError("Unable to find native structures")
    paths.processed.mkdir(exist_ok=True)

    return args, paths


@logger.catch()
def main():
    args, paths = parse_args()

    # These steps are common for both training and evaluation:
    # - primary sequences
    # - DSSP features
    # - multiple-sequence alignments
    fasta_wrapper = FastaToNumpyWrapper(paths.sequences)
    run_dssp(
        input_dir_or_tar=paths.decoys,
        output_dir=paths.decoys,
        threads=args.cpus,
        verbose=0,
    )
    run_msa(paths.sequences, args.uniref, paths.alignments, args.cpus)

    # Useful metadata
    msa_counts_dict = pd.read_pickle(paths.alignments)
    target_ids = list(msa_counts_dict.keys())

    # These steps are only needed for training
    # - TM, GDT_TS, GDT_HA scores (global)
    # - CAD score (global and local)
    # - LDDT score (global and local)
    if args.train:
        for target_id in target_ids:
            native_pdb = paths.native.joinpath(target_id).with_suffix(".pdb")
            if not native_pdb.is_file():
                logger.error(f"Can not find: {native_pdb}")
                continue

            decoys_dir = paths.decoys / target_id
            if not decoys_dir.is_dir():
                logger.error(f"Can not find: {decoys_dir}")
                continue

            output_npz = paths.decoys.joinpath(target_id)

            run_tmscore(
                native_pdb,
                decoys_dir,
                output_npz.with_suffix(".tmscore.npz"),
                tmscore=args.tmscore,
            )

            run_cadscore(
                native_pdb,
                decoys_dir,
                output_npz.with_suffix(".cad.npz"),
                sequence_length=msa_counts_dict[target_id].shape[0],
                voronota=args.voronota,
            )

            run_lddt(
                native_pdb,
                decoys_dir,
                output_npz.with_suffix(".lddt.npz"),
                sequence_length=msa_counts_dict[target_id].shape[0],
            )

    # Finally, build a graph data structure for PyTorch
    pgb = ProteinGraphBuilder(max_distance=12, max_hops=5)
    for target_id in target_ids:
        logger.info(f"Processing {target_id}")
        decoys_dir = paths.decoys / target_id
        if not decoys_dir.is_dir():
            logger.error(f"Can not find: {decoys_dir}")
            continue

        sequence_np = fasta_wrapper[target_id]
        df_msa_feats = compute_msa_features(msa_counts_dict[target_id])

        target_dict = {
            "dataset_id": paths.root.name,
            "target_id": target_id,
            "sequence": torch.from_numpy(sequence_np),
            "msa_feats": torch.from_numpy(df_msa_feats.values).float(),
            "graphs": [],
        }

        for decoy_pdb in decoys_dir.glob("*.pdb"):
            decoy_id = decoy_pdb.with_suffix("").name
            decoy_dssp = decoy_pdb.with_suffix(".dssp")
            df_decoy_feats = load_decoy_feats(decoy_pdb, decoy_dssp, len(sequence_np))

            if args.train:
                df_qa_global, df_qa_local = load_df_qa(paths.decoys.joinpath(target_id))
            else:
                df_qa_local = None
                df_qa_global = None

            graph = pgb.build(decoy_id, df_decoy_feats, df_qa_global, df_qa_local)
            graph.debug()
            target_dict["graphs"].append(graph)

        torch.save(target_dict, paths.processed / f"{target_id}.pth")


def load_df_qa(output_npz: Path):
    lddt_dict = np.load(output_npz.with_suffix(".lddt.npz").as_posix())
    global_lddt = pd.Series(
        lddt_dict["global_lddt"], index=lddt_dict["decoys"], name="lddt"
    )
    local_lddt = pd.DataFrame(lddt_dict["local_lddt"], index=lddt_dict["decoys"])

    tmscore_dict = np.load(output_npz.with_suffix(".tmscore.npz").as_posix())
    df_tmscore = pd.DataFrame({**tmscore_dict}).set_index("decoys")

    cad_dict = np.load(output_npz.with_suffix(".cad.npz").as_posix())
    global_cad = pd.Series(cad_dict["global_cad"], index=cad_dict["decoys"], name="cad")
    local_cad = pd.DataFrame(cad_dict["local_cad"], index=cad_dict["decoys"])

    df_global = pd.concat((df_tmscore, global_lddt, global_cad), axis=1).rename_axis(
        index="decoy_id"
    )

    df_local = (
        pd.concat(
            (local_lddt.stack(dropna=False), local_cad.stack(dropna=False)),
            axis=1,
            keys=["lddt", "cad"],
            join="inner",
        )
            .sort_index()
            .rename_axis(index=["decoy_id", "residue_idx"])
    )

    return df_global, df_local


if __name__ == "__main__":
    main()
