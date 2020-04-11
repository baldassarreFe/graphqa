from typing import Tuple

import numpy as np
import torch
import torch_scatter
from omegaconf import OmegaConf
from torch_geometric.data import Batch
from torch.nn import (
    Module,
    Sequential,
    Linear,
    ReLU,
    Sigmoid,
    Embedding,
    ModuleDict,
    ModuleList,
    Dropout,
    BatchNorm1d,
    Identity,
)


class GraphQA(Module):
    def __init__(self, conf: OmegaConf):
        super().__init__()

        # Configuration
        mp_in_edge_feats = (
            conf.encoder.out_edge_feats + conf.embeddings.sep + conf.embeddings.rbf
        )
        mp_in_node_feats = (
            conf.encoder.out_node_feats + conf.embeddings.aa + conf.embeddings.ss
        )

        mp_edge_feats = layer_sizes_exp2(
            mp_in_edge_feats, conf.mp.out_edge_feats, conf.mp.layers, round_pow2=True
        )
        mp_node_feats = layer_sizes_exp2(
            mp_in_node_feats, conf.mp.out_node_feats, conf.mp.layers, round_pow2=True
        )
        mp_global_feats = layer_sizes_exp2(
            conf.mp.in_global_feats,
            conf.mp.out_global_feats,
            conf.mp.layers,
            round_pow2=True,
        )
        mp_sizes = zip(mp_edge_feats, mp_node_feats, [0] + mp_global_feats[1:])

        self.readout_concat = conf.readout.concat

        # Embeddings (aa type, dssp classification, separation, distance)
        self.embeddings = ModuleDict(
            {
                "amino_acid": Embedding(
                    num_embeddings=20, embedding_dim=conf.embeddings.aa
                ),
                "secondary_structure": Embedding(
                    num_embeddings=9, embedding_dim=conf.embeddings.ss
                ),
                "separation": SeparationEmbedding(
                    bins=(1, 2, 3, 4, 5, 10, 15), embedding_dim=conf.embeddings.sep
                ),
                "distance_rbf": RbfDistanceEncoding(
                    min_dist=0, max_dist=20, num_bases=conf.embeddings.rbf
                ),
            }
        )

        # Encoder (dssp features on the nodes and geometric features on the edges)
        self.encoder = Encoder(
            out_edge_feats=conf.encoder.out_edge_feats,
            out_node_feats=conf.encoder.out_node_feats,
        )

        # Message passing
        self.message_passing = ModuleList()
        in_e, in_n, in_g = next(mp_sizes)
        for out_e, out_n, out_g in mp_sizes:
            mp = MessagePassing(
                in_edge_feats=in_e,
                in_node_feats=in_n,
                in_global_feats=in_g,
                out_edge_feats=out_e,
                out_node_feats=out_n,
                out_global_feats=out_g,
                dropout=conf.mp.dropout,
                batch_norm=conf.mp.batch_norm,
            )
            self.message_passing.append(mp)
            in_e, in_n, in_g = out_e, out_n, out_g

        # Readout
        if self.readout_concat:
            in_n += mp_in_node_feats
        self.readout = Readout(in_n, in_g)

    @staticmethod
    def prepare(graphs: Batch) -> Tuple[torch.Tensor, ...]:
        aa = graphs.aa
        msa_feats = graphs.msa_feats
        x = graphs.x
        edge_index = graphs.edge_index
        edge_attr = graphs.edge_attr
        secondary_structure = graphs.secondary_structure
        batch = graphs.batch

        return aa, msa_feats, x, edge_index, edge_attr, secondary_structure, batch

    def forward(
        self, aa, msa_feats, x, edge_index, edge_attr, secondary_structure, batch
    ):
        # Embeddings (aa type, dssp classification, separation, distance)
        aa = self.embeddings.amino_acid(aa.long())
        ss = self.embeddings.secondary_structure(secondary_structure.long())
        sep = self.embeddings.separation(edge_index)
        rbf = self.embeddings.distance_rbf(edge_attr[:, 0])

        # Encoder (dssp features on the nodes and geometric features on the edges)
        x, edge_attr = self.encoder(x, msa_feats, edge_attr)

        # Message passing
        x = x_mp = torch.cat((aa, x, ss), dim=1)
        edge_attr = torch.cat((edge_attr, sep, rbf), dim=1)
        num_graphs = batch[-1].item() + 1
        u = torch.empty(num_graphs, 0, dtype=torch.float, device=x.device)
        for mp in self.message_passing:
            x, edge_attr, edge_index, u, batch = mp(x, edge_attr, edge_index, u, batch)

        # Readout
        if self.readout_concat:
            x = torch.cat((x, x_mp), dim=1)
        x, u = self.readout(x, u)
        return x, u


class SeparationEmbedding(Module):
    def __init__(self, embedding_dim, bins: tuple):
        super().__init__()
        self.bins = bins
        self.emb = Embedding(num_embeddings=len(bins) + 1, embedding_dim=embedding_dim)

    @torch.jit.ignore
    def _sep_to_code(self, separation):
        codes = np.digitize(separation.abs().cpu().numpy(), bins=self.bins, right=True)
        codes = torch.from_numpy(codes).to(separation.device)
        return codes

    def forward(self, edge_index):
        separation = edge_index[0] - edge_index[1]
        codes = self._sep_to_code(separation)
        embeddings = self.emb(codes)
        return embeddings


class RbfDistanceEncoding(Module):
    def __init__(self, min_dist: float, max_dist: float, num_bases: int):
        super().__init__()
        if not 0 <= min_dist < max_dist:
            raise ValueError(
                f"Invalid RBF centers: 0 <= {min_dist} < {max_dist} is False"
            )
        if num_bases < 0:
            raise ValueError(f"Invalid RBF size: 0 < {num_bases} is False")
        self.register_buffer(
            "rbf_centers", torch.linspace(min_dist, max_dist, steps=num_bases)
        )

    def forward(self, distances):
        # assert distances.ndim == 1
        # Distances are encoded using a equally spaced RBF kernels with unit variance
        rbf = torch.exp(-((distances[:, None] - self.rbf_centers[None, :]) ** 2))
        return rbf

    def extra_repr(self):
        return f"bases={len(self.rbf_centers)}"


class Encoder(Module):
    def __init__(self, out_edge_feats, out_node_feats):
        super().__init__()
        self.node_encoder = Sequential(
            Linear(3 + 21, out_node_feats // 2),
            ReLU(),
            Linear(out_node_feats // 2, out_node_feats),
            ReLU(),
        )
        self.edge_encoder = Sequential(
            Linear(4, out_edge_feats // 2),
            ReLU(),
            Linear(out_edge_feats // 2, out_edge_feats),
            ReLU(),
        )

    def forward(self, x, msa_feats, edge_attr):
        x = self.node_encoder(torch.cat((x, msa_feats), dim=1))
        edge_attr = self.edge_encoder(edge_attr)
        return x, edge_attr


class MessagePassing(Module):
    def __init__(
        self,
        in_edge_feats: int,
        in_node_feats: int,
        in_global_feats: int,
        out_edge_feats: int,
        out_node_feats: int,
        out_global_feats: int,
        batch_norm: bool,
        dropout: float,
    ):
        super().__init__()
        in_feats = in_node_feats + in_edge_feats + in_global_feats
        self.edge_fn = Sequential(
            Linear(in_feats, out_edge_feats),
            Dropout(p=dropout) if dropout > 0 else Identity(),
            BatchNorm1d(out_edge_feats) if batch_norm else Identity(),
            ReLU(),
        )
        in_feats = in_node_feats + out_edge_feats + in_global_feats
        self.node_fn = Sequential(
            Linear(in_feats, out_node_feats),
            Dropout(p=dropout) if dropout > 0 else Identity(),
            BatchNorm1d(out_node_feats) if batch_norm else Identity(),
            ReLU(),
        )
        in_feats = out_node_feats + out_edge_feats + in_global_feats
        self.global_fn = Sequential(
            Linear(in_feats, out_global_feats),
            Dropout(p=dropout) if dropout > 0 else Identity(),
            BatchNorm1d(out_global_feats) if batch_norm else Identity(),
            ReLU(),
        )

    def forward(self, x, edge_attr, edge_index, u, batch):
        x_src = x[edge_index[0]]
        u_src = u[batch[edge_index[0]]]
        edge_attr = torch.cat((x_src, edge_attr, u_src), dim=1)
        edge_attr = self.edge_fn(edge_attr)

        msg_to_node = torch_scatter.scatter(
            edge_attr, edge_index[1], dim=0, dim_size=x.shape[0], reduce="mean"
        )
        u_to_node = u[batch]
        x = torch.cat((x, msg_to_node, u_to_node), dim=1)
        x = self.node_fn(x)

        edge_global = torch_scatter.scatter(
            edge_attr, batch[edge_index[0]], dim=0, dim_size=u.shape[0], reduce="mean"
        )
        x_global = torch_scatter.scatter(
            x, batch, dim=0, dim_size=u.shape[0], reduce="mean"
        )
        u = torch.cat((edge_global, x_global, u), dim=1)
        u = self.global_fn(u)

        return x, edge_attr, edge_index, u, batch


class Readout(Module):
    def __init__(self, in_node_feats, in_global_feats):
        super().__init__()
        self.node_fn = Sequential(Linear(in_node_feats, 2), Sigmoid())
        self.global_fn = Sequential(Linear(in_global_feats, 5), Sigmoid())

    def forward(self, x, u):
        x = self.node_fn(x)
        u = self.global_fn(u)
        return x, u


def round_to_pow2(value):
    return np.exp2(np.round(np.log2(value))).astype(int)


def layer_sizes_linear(in_feats, out_feats, layers, round_pow2=False):
    sizes = np.linspace(in_feats, out_feats, layers + 1).round().astype(np.int)
    if round_pow2:
        sizes[1:-1] = round_to_pow2(sizes[1:-1])
    return sizes.tolist()


def layer_sizes_exp2(in_feats, out_feats, layers, round_pow2=False):
    sizes = (
        np.logspace(np.log2(in_feats), np.log2(out_feats), layers + 1, base=2)
        .round()
        .astype(np.int)
    )
    if round_pow2:
        sizes[1:-1] = round_to_pow2(sizes[1:-1])
    return sizes.tolist()
