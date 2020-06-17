import argparse
import multiprocessing
from pathlib import Path

import torch
import pandas as pd
from loguru import logger

import graphqa.data.msa as msa
import graphqa.data.dssp as dssp
from graphqa.data.paths import dataset_paths
from .decoys import load_decoy_feats
from .graphs import ProteinGraphBuilder
from .sequences import FastaToNumpyWrapper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", help="Data dir.")
    parser.add_argument("uniref", help="Uniref database.")
    # TODO add argument --skip-existing
    args = parser.parse_args()
    paths = dataset_paths(args.dataset_dir)

    fasta_wrapper = FastaToNumpyWrapper(paths.sequences)
    msa.main(paths.sequences, args.uniref, paths.alignments, max(1, multiprocessing.cpu_count()-1))
    dssp.dssp(paths.decoys, paths.decoys, threads=4, verbose=0)

    # TODO if natives are present compute ground truth scores,
    #      this is only useful for training, not for inference

    pgb = ProteinGraphBuilder(max_distance=12, max_hops=5)
    msa_counts_dict = pd.read_pickle(paths.alignments)
    paths.processed.mkdir(exist_ok=True)

    for target_dir in filter(Path.is_dir, paths.decoys.iterdir()):
        target_id = target_dir.name
        logger.info(f'Processing {target_id}')
        sequence_np = fasta_wrapper[target_id]
        df_msa_feats = msa.compute_features(msa_counts_dict[target_id])

        target_dict = {
            "dataset_id": paths.root.name,
            "target_id": target_id,
            "sequence": torch.from_numpy(sequence_np),
            "msa_feats": torch.from_numpy(df_msa_feats.values).float(),
            "graphs": [],
        }

        # TODO load ground-truth scores if present and pass them to pgb.build,
        #      this is only useful for training, not for inference
        #      (target_id.{cad,lddt,tmscore}.npz)

        for decoy_pdb in target_dir.glob("*.pdb"):
            decoy_id = decoy_pdb.with_suffix("").name
            decoy_dssp = decoy_pdb.with_suffix(".dssp")
            df_decoy_feats = load_decoy_feats(decoy_pdb, decoy_dssp, len(sequence_np))
            graph = pgb.build(decoy_id, df_decoy_feats)
            graph.debug()
            target_dict["graphs"].append(graph)

        torch.save(target_dict, paths.processed / f"{target_id}.pth")


if __name__ == "__main__":
    main()
