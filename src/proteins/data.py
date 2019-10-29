from __future__ import annotations

from typing import Sequence

import torch
import torchgraphs as tg


class Decoy(tg.Graph):
    _feature_fields = tg.Graph._feature_fields + ('residues', 'lddt', 'gdtts', 'distances')

    def __init__(
            self,
            target_name: str,
            decoy_name: str,
            residues: torch.LongTensor,
            senders: torch.LongTensor,
            receivers: torch.LongTensor,
            edge_features: torch.Tensor,
            node_features: torch.Tensor,
            distances: torch.Tensor,
            lddt: torch.Tensor,
            gdtts: torch.Tensor,
    ):
        super(Decoy, self).__init__(
            num_nodes=len(residues),
            node_features=node_features,
            edge_features=edge_features,
            senders=senders,
            receivers=receivers
        )
        self.target_name = target_name
        self.decoy_name = decoy_name
        self.residues = residues
        self.distances = distances
        self.lddt = lddt
        self.gdtts = gdtts

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"{self.target_name}/{self.decoy_name}, "
                f"n={self.num_nodes}, "
                f"e={self.num_edges}, "
                f"n_shape={tuple(self.node_features_shape)}" +
                (f"e_shape={tuple(self.edge_features_shape)}" if self.edge_features is not None else "") +
                f")")


class DecoyBatch(tg.GraphBatch):
    _feature_fields = tg.GraphBatch._feature_fields + ('residues', 'lddt', 'gdtts', 'distances')

    def __init__(
            self,
            target_name,
            decoy_name,
            residues,
            distances,
            lddt,
            gdtts,
            num_nodes=None,
            num_graphs=None,
            senders=None,
            receivers=None,
            num_nodes_by_graph=None,
            num_edges_by_graph=None,
            node_features=None,
            edge_features=None,
            node_index_by_graph=None,
            edge_index_by_graph=None,
            global_features=None,
    ):
        super(DecoyBatch, self).__init__(
            num_nodes=num_nodes,
            num_graphs=num_graphs,
            senders=senders,
            receivers=receivers,
            num_nodes_by_graph=num_nodes_by_graph,
            num_edges_by_graph=num_edges_by_graph,
            node_features=node_features,
            edge_features=edge_features,
            node_index_by_graph=node_index_by_graph,
            edge_index_by_graph=edge_index_by_graph,
            global_features=global_features,
        )
        self.target_name = target_name
        self.decoy_name = decoy_name
        self.residues = residues
        self.distances = distances
        self.lddt = lddt
        self.gdtts = gdtts

    @classmethod
    def from_graphs(cls, graphs: Sequence[Decoy]) -> DecoyBatch:
        """Merges multiple decoys in a batch
        """
        if len(graphs) == 0:
            raise ValueError('Graphs list can not be empty')

        node_features = []
        edge_features = []
        distances = []
        num_nodes_by_graph = []
        num_edges_by_graph = []
        senders = []
        receivers = []
        target_name = []
        decoy_name = []
        residues = []
        lddt = []
        gdtts = []
        node_offset = 0
        for i, g in enumerate(graphs):
            target_name.append(g.target_name)
            decoy_name.append(g.decoy_name)
            residues.append(g.residues)
            lddt.append(g.lddt)
            gdtts.append(g.gdtts)
            node_features.append(g.node_features)
            edge_features.append(g.edge_features)
            distances.append(g.distances)
            num_nodes_by_graph.append(g.num_nodes)
            num_edges_by_graph.append(g.num_edges)
            senders.append(g.senders + node_offset)
            receivers.append(g.receivers + node_offset)
            node_offset += g.num_nodes

        use_shared_memory = torch.utils.data.get_worker_info() is not None

        out = None
        if use_shared_memory:
            numel = sum([x.numel() for x in node_features])
            storage = node_features[0].storage()._new_shared(numel)
            out = node_features[0].new(storage)
        node_features = torch.cat(node_features, out=out)

        out = None
        if use_shared_memory:
            numel = sum([x.numel() for x in edge_features])
            storage = edge_features[0].storage()._new_shared(numel)
            out = edge_features[0].new(storage)
        edge_features = torch.cat(edge_features, out=out)

        out = None
        if use_shared_memory:
            numel = sum([x.numel() for x in distances])
            storage = distances[0].storage()._new_shared(numel)
            out = distances[0].new(storage)
        distances = torch.cat(distances, out=out)

        out = None
        if use_shared_memory:
            numel = sum([x.numel() for x in lddt])
            storage = graphs[0].lddt.storage()._new_shared(numel)
            out = lddt[0].new(storage)
        lddt = torch.cat(lddt, out=out)

        out = None
        if use_shared_memory:
            numel = sum([x.numel() for x in gdtts])
            storage = graphs[0].gdtts.storage()._new_shared(numel)
            out = gdtts[0].new(storage)
        gdtts = torch.cat(gdtts, out=out)

        out = None
        if use_shared_memory:
            numel = sum([x.numel() for x in residues])
            storage = graphs[0].residues.storage()._new_shared(numel)
            out = residues[0].new(storage)
        residues = torch.cat(residues, out=out)

        out = None
        if use_shared_memory:
            numel = sum([x.numel() for x in senders])
            storage = graphs[0].senders.storage()._new_shared(numel)
            out = senders[0].new(storage)
        senders = torch.cat(senders, out=out)

        out = None
        if use_shared_memory:
            numel = sum([x.numel() for x in receivers])
            storage = graphs[0].receivers.storage()._new_shared(numel)
            out = receivers[0].new(storage)
        receivers = torch.cat(receivers, out=out)

        return cls(
            target_name=target_name,
            decoy_name=decoy_name,
            residues=residues,
            lddt=lddt,
            gdtts=gdtts,
            num_nodes=node_offset,
            num_nodes_by_graph=senders.new_tensor(num_nodes_by_graph),
            num_edges_by_graph=senders.new_tensor(num_edges_by_graph),
            node_features=node_features,
            edge_features=edge_features,
            distances=distances,
            senders=senders,
            receivers=receivers
        )

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"#{self.num_graphs}, "
                f"n={self.num_nodes}, "
                f"e={self.num_edges}, "
                f"n_shape={tuple(self.node_features_shape)}" +
                (f"e_shape={tuple(self.edge_features_shape)}" if self.edge_features is not None else "") +
                f")")
