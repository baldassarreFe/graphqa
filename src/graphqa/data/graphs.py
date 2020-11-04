from typing import Optional

import torch
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph
from torch_geometric.transforms import PointPairFeatures


class ProteinGraphBuilder(object):
    """
    Builds a protein graph in pytorch format in the following steps:

    - radius_graph connects residues based on their spatial location (distance in angstroms)
    - backbone connects residues based on their separation in the chain (hops)
    - geometric_feats computes distances and angles
    - fillna fixes missing dssp
    """

    def __init__(self, max_distance=12, max_hops=5):
        # max_num_neighbors is large enough that usually no neighbor is excluded (if max_distance is reasonable)
        self.radius_graph = RadiusGraphWithNans(
            max_distance, loop=False, max_num_neighbors=128
        )
        self.backbone = SequentialEdges(hops=max_hops)
        self.geometric_feats = PointPairFeatures()
        self.fillna = FillNa()
        self.cleanup = CleanUp()

    def build(
        self,
        decoy_id: str,
        decoy_feats: pd.DataFrame,
        qa_global: Optional[pd.DataFrame] = None,
        qa_local: Optional[pd.DataFrame] = None,
    ) -> Data:
        # Input features
        graph = Data(
            # Metadata
            decoy_id=decoy_id,
            num_nodes=len(decoy_feats),
            # Node features
            x=torch.from_numpy(
                decoy_feats["dssp"][["surface_acc", "phi", "psi"]].values
            ).float(),
            # Categorical features
            secondary_structure=torch.from_numpy(decoy_feats["dssp"]["ss"].values),
            # Spatial features
            pos=torch.from_numpy(decoy_feats["coords_ca"].values).float(),
            norm=torch.from_numpy(decoy_feats["orient_res"].values).float(),
        )

        # Geometric transforms
        self.radius_graph(graph)
        self.backbone(graph)
        self.geometric_feats(graph)
        self.fillna(graph)
        self.cleanup(graph)

        # QA targets
        if qa_global is not None:
            qa_global = qa_global[
                ["tm_score", "gdt_ts", "gdt_ha", "lddt", "cad"]
            ].values
            graph.qa_global = torch.from_numpy(qa_global).float()[None, :]
        if qa_local is not None:
            qa_local = qa_local[["lddt", "cad"]].values
            graph.qa_local = torch.from_numpy(qa_local).float()

        return graph


class RadiusGraphWithNans(object):
    def __init__(self, r, loop=False, max_num_neighbors=32):
        self.r = r
        self.loop = loop
        self.max_num_neighbors = max_num_neighbors

    def __call__(self, data: Data) -> Data:
        data.edge_attr = None
        nans = torch.isnan(data.pos[:, 0])
        pos = data.pos.clone()
        pos[nans] = 10_000
        batch = data.batch if "batch" in data else None

        edge_index = radius_graph(
            pos, self.r, batch, self.loop, self.max_num_neighbors,
        )
        na_sender = torch.any(edge_index[0] == torch.nonzero(nans, as_tuple=False), dim=0)
        na_receiver = torch.any(edge_index[0] == torch.nonzero(nans, as_tuple=False), dim=0)
        drop = na_sender | na_receiver

        data.edge_index = edge_index[:, ~drop]
        return data


class SequentialEdges(object):
    """
    Add H-hop edges to a linear graph.
    For a linear graph with N nodes, each hop of distance h adds 2*(N-h) directed edges.
    Adding all edges from h=1 to h=H gives H*(2*N-H-1) edges.
    """

    def __init__(self, hops=3):
        self.hops = tuple(range(1, hops + 1))

    def __call__(self, data: Data) -> Data:
        if "batch" in data:
            raise ValueError("Doesn't work with batched graphs")
        if "edge_attr" in data:
            raise ValueError("Can't add new edges to a graph with edge attributes")

        forward_hop_edges = torch.tensor(
            [(i, i + hop) for hop in self.hops for i in range(data.num_nodes - hop)]
        ).transpose(0, 1)
        backward_hop_edges = forward_hop_edges[[1, 0], :]

        if data.edge_index is None:
            data.edge_index = torch.cat((forward_hop_edges, backward_hop_edges), dim=1)
        else:
            data.edge_index = torch.cat(
                (data.edge_index, forward_hop_edges, backward_hop_edges), dim=1
            )

        return data.coalesce()


class FillNa(object):
    def __call__(self, data: Data) -> Data:
        data.x = torch.where(torch.isnan(data.x), torch.tensor(0.0), data.x)
        data.edge_attr = torch.where(
            torch.isnan(data.edge_attr), torch.tensor(0.0), data.edge_attr
        )
        return data


class CleanUp(object):
    def __call__(self, data: Data) -> Data:
        data.pos = None
        data.norm = None
        return data
