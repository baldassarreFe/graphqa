import functools
from collections import defaultdict
from pathlib import Path
from typing import Tuple, Dict, Iterator, Optional, Union, Sized, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from torch_geometric.data import Data

from loguru import logger


class DecoyDataset(Dataset):
    def __init__(self, pth_paths: Sequence[Path], transforms=None):
        super().__init__()

        self.targets_by_target_id: Dict[str, Data] = {}
        self.targets_by_casp_ed_and_target_id: Dict[Tuple[int, str], Data] = {}
        self.decoys_by_target_id_and_decoy_id: Dict[Tuple[str, str], Data] = {}
        self.decoys_by_ds_ed_and_target_id_and_decoy_id: Dict[
            Tuple[int, str, str], Data
        ] = {}

        logger.debug(f"Starting to load graphs from {len(pth_paths)} pth files")
        skipped = 0
        for p in pth_paths:
            target = torch.load(p)
            dataset_id = target.get("casp_ed", target.get("dataset_id"))
            target_id = target["target_id"]

            self.targets_by_target_id[target_id] = target
            self.targets_by_casp_ed_and_target_id[(dataset_id, target_id)] = target

            for decoy in target["graphs"]:
                decoy_id = decoy.decoy_id

                # Skip loading if the decoy:
                # - contains all NaN values for one of its local scores
                # - contains all NaN values for its global scores
                if hasattr(decoy, "qa_local") and torch.isnan(decoy.qa_local).all(
                    dim=0, keepdim=True
                ).any(dim=1):
                    logger.warning(
                        f"Decoy {target_id}/{decoy_id} contains all NaN values "
                        f"in one of the columns of ground-truth local scores"
                    )
                    skipped += 1
                    continue
                if hasattr(decoy, "qa_global") and torch.isnan(decoy.qa_global).all(
                    dim=1
                ):
                    logger.warning(
                        f"Decoy {target_id}/{decoy_id} contains all NaN values "
                        f"for its ground-truth global scores"
                    )
                    skipped += 1
                    continue

                self.add_target_feats_to_decoy(target, decoy)
                self.decoys_by_target_id_and_decoy_id[(target_id, decoy_id)] = decoy
                self.decoys_by_ds_ed_and_target_id_and_decoy_id[
                    (dataset_id, target_id, decoy_id)
                ] = decoy

        if len(self) == 0:
            logger.warning("Empty dataset!")
        else:
            logger.debug(f"Done loading {len(self)} graphs, skipped {skipped}")
        self.transforms = transforms

    @staticmethod
    def add_target_feats_to_decoy(target: dict, decoy: Data):
        """
        Some features are shared between all decoys of a target,
        e.g. casp_ed, target_id, aa sequence, msa_features.
        Args:
            target:
            decoy:

        Returns:

        """
        decoy.casp_ed = target.get("casp_ed", target.get("dataset_id"))
        decoy.target_id = target["target_id"]
        decoy.n_nodes = decoy.num_nodes
        decoy.n_edges = decoy.num_edges
        decoy.msa_feats = target["msa_feats"]
        decoy.aa = target["sequence"]

        if decoy.num_nodes == 0:
            # If a graph has 0 nodes and it's put last in the the batch formed by
            # Batch.from_data_list(...) it will cause a miscount in batch.num_graphs
            logger.warning(
                f"Found graph with 0 nodes: {decoy.casp_ed}/{decoy.target_id}/{decoy.decoy_id}"
            )

    def __getitem__(self, item):
        if isinstance(item, int):
            item = self.keys[item]
        *casp_ed, target_id, decoy_id = item
        graph = self.decoys_by_target_id_and_decoy_id[(target_id, decoy_id)]
        if self.transforms is not None:
            graph = self.transforms(graph.clone())
        return graph

    def __len__(self):
        return len(self.decoys_by_ds_ed_and_target_id_and_decoy_id)

    @functools.cached_property
    def casp_editions(self) -> Tuple[int]:
        return tuple(set(k[0] for k in self.targets_by_casp_ed_and_target_id.keys()))

    @functools.cached_property
    def target_ids(self) -> Tuple[str]:
        return tuple(self.targets_by_target_id.keys())

    @functools.cached_property
    def keys(self) -> Tuple[Tuple[int, str, str], ...]:
        return tuple(self.decoys_by_ds_ed_and_target_id_and_decoy_id.keys())


# TODO deprecate this function
def iter_pth_files(
    data_dir: Union[str, Path],
    casp_ed: Optional[str] = None,
    target_id: Optional[str] = None,
) -> Iterator[Path]:
    if casp_ed is None:
        casp_ed = "*"
    if target_id is None:
        target_id = "*"
    data_dir = Path(data_dir).expanduser().resolve()
    return data_dir.glob(f"CASP{casp_ed}/processed/{target_id}.pth")


class RandomTargetSampler(Sampler):
    """
    Sample decoys from a dataset:
    targets are randomly sorted,
    then for each target one decoy is sampled at random.
    """

    def __init__(self, dataset: DecoyDataset, rg: np.random.Generator):
        super().__init__(dataset)
        self.rg = rg
        self.target_ids = dataset.target_ids
        self.decoys_by_target = defaultdict(list)
        for casp_ed, target_id, decoy_id in dataset.keys:
            self.decoys_by_target[target_id].append((casp_ed, target_id, decoy_id))
        assert len(self.decoys_by_target) == len(self.target_ids)

    def __iter__(self):
        target_ids = self.rg.permutation(self.target_ids)
        for target_id in target_ids:
            decoys_list = self.decoys_by_target[target_id]
            yield self.rg.choice(decoys_list)

    def __len__(self):
        return len(self.decoys_by_target)
