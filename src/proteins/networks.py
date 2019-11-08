from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np

import torchgraphs as tg
from torchsearchsorted import searchsorted

from proteins.data import DecoyBatch
from . import features


class ProteinGN(nn.Module):
    def __init__(
            self,
            min_dist=0,
            max_dist=20,
            rbf_size=16,
            residue_emb_size=32,
            separation_enc='categorical',
            distance_enc='rbf',
            enc_in_nodes=83,
            enc_in_edges=8,
            layers=1,
            mp_in_edges=8,
            mp_in_nodes=16,
            mp_in_globals=4,
            mp_out_edges=64,
            mp_out_nodes=128,
            mp_out_globals=32,
            dropout=0,
            batch_norm=False,
    ):
        super().__init__()

        preprocessing = OrderedDict()

        # Spatial distances
        if distance_enc == 'absent':
            pass
        elif distance_enc == 'scalar':
            preprocessing['distance_encoding'] = ScalarDistanceEncodingEdges()
        elif distance_enc == 'rbf':
            preprocessing['distance_encoding'] = RbfDistanceEncodingEdges(min_dist, max_dist, rbf_size)
        else:
            raise ValueError(f'Invalid `distance_enc`: {distance_enc}')

        # Sequential distances
        if separation_enc == 'absent':
            pass
        elif separation_enc == 'scalar':
            preprocessing['separation_encoding'] = ScalarSeparationEncodingEdges()
        elif separation_enc == 'categorical':
            preprocessing['separation_encoding'] = CategoricalSeparationEncodingEdges()
        else:
            raise ValueError(f'Invalid `separation_enc`: {separation_enc}')

        # Embedding of the amino acid sequence
        preprocessing['residue_embedding'] = ResidueEmbedding(residue_emb_size)

        # Duplicate edges
        preprocessing['duplicate_edges'] = DuplicateEdges()

        self.preprocessing = nn.Sequential(preprocessing)

        # Edge feature shape: E -> mp_in_edges//2 -> mp_in_edges
        # Node feature shape: N -> mp_in_nodes//2 -> mp_in_nodes
        # Global feature shape: None -> mp_in_globals
        self.encoder = nn.Sequential(OrderedDict({
            'edge1': tg.EdgeLinear(out_features=mp_in_edges//2, edge_features=enc_in_edges if enc_in_edges > 0 else None),
            'edge1_dropout': tg.EdgeDropout(p=dropout) if dropout > 0 else nn.Identity(),
            'edge1_bn': tg.EdgeBatchNorm(num_features=mp_in_edges//2) if batch_norm else nn.Identity(),
            'edge1_relu': tg.EdgeReLU(),
            'edge2': tg.EdgeLinear(out_features=mp_in_edges, edge_features=mp_in_edges//2),
            'edge2_dropout': tg.EdgeDropout(p=dropout),
            'edge2_bn': tg.EdgeBatchNorm(num_features=mp_in_edges) if batch_norm else nn.Identity(),
            'edge2_relu': tg.EdgeReLU(),

            'node1': tg.NodeLinear(out_features=mp_in_nodes//2, node_features=enc_in_nodes),
            'node1_dropout': tg.NodeDropout(p=dropout) if dropout > 0 else nn.Identity(),
            'node1_bn': tg.NodeBatchNorm(num_features=mp_in_nodes//2) if batch_norm else nn.Identity(),
            'node1_relu': tg.NodeReLU(),
            'node2': tg.NodeLinear(out_features=mp_in_nodes, node_features=mp_in_nodes//2),
            'node2_dropout': tg.NodeDropout(p=dropout) if dropout > 0 else nn.Identity(),
            'node2_bn': tg.NodeBatchNorm(num_features=mp_in_nodes) if batch_norm else nn.Identity(),
            'node2_relu': tg.NodeReLU(),

            'global': tg.GlobalLinear(out_features=mp_in_globals, bias=True),
            'global_dropout': tg.GlobalDropout(p=dropout) if dropout > 0 else nn.Identity(),
            'global_relu': tg.GlobalReLU(),
        }))

        # Message passing layers: edge, node and global shapes go
        # from (mp_in_edges , mp_in_nodes , mp_in_globals )
        # to   (mp_out_edges, mp_out_nodes, mp_out_globals)
        # following powers of 2 in the number of steps given with the `layers` parameter (e.g. 10).

        mp_layers_output_sizes = np.power(2, np.linspace(
            np.log2((mp_in_edges, mp_in_nodes, mp_in_globals)),
            np.log2((mp_out_edges, mp_out_nodes, mp_out_globals)),
            num=layers, endpoint=False
        ).round().astype(int)).tolist()[1:] + [(mp_out_edges, mp_out_nodes, mp_out_globals)]

        layers_ = []
        in_e, in_n, in_g = mp_in_edges, mp_in_nodes, mp_in_globals
        for out_e, out_n, out_g in mp_layers_output_sizes:
            layer = nn.Sequential(OrderedDict({
                'edge1': tg.EdgeLinear(
                    out_features=out_e,
                    edge_features=in_e,
                    sender_features=in_n,
                    global_features=in_g
                ),
                'edge1_dropout': tg.EdgeDropout(p=dropout) if dropout > 0 else nn.Identity(),
                'edge1_bn': tg.EdgeBatchNorm(num_features=out_e) if batch_norm else nn.Identity(),
                'edge1_relu': tg.EdgeReLU(),

                'node1': tg.NodeLinear(
                    out_features=out_n,
                    node_features=in_n,
                    incoming_features=out_e,
                    global_features=in_g,
                    aggregation='mean',
                ),
                'node1_dropout': tg.NodeDropout(p=dropout) if dropout > 0 else nn.Identity(),
                'node1_bn': tg.NodeBatchNorm(num_features=out_n) if batch_norm else nn.Identity(),
                'node1_relu': tg.NodeReLU(),

                'global1': tg.GlobalLinear(
                    out_features=out_g,
                    edge_features=out_e,
                    node_features=out_n,
                    global_features=in_g,
                    aggregation='mean',
                ),
                'global1_dropout': tg.GlobalDropout(p=dropout) if dropout > 0 else nn.Identity(),
                'global1_bn': tg.GlobalBatchNorm(num_features=out_g) if batch_norm else nn.Identity(),
                'global1_relu': tg.GlobalReLU(),
            }))
            layers_.append(layer)
            in_e, in_n, in_g = out_e, out_n, out_g

        self.layers = torch.nn.Sequential(OrderedDict({f'layer_{i}': l for i, l in enumerate(layers_)}))

        # Node feature shape: mp_out_nodes -> 1
        # Global feature shape: mp_out_globals -> 1
        self.readout = nn.Sequential(OrderedDict({
            'node': tg.NodeLinear(features.Output.Node.LENGTH, node_features=mp_out_nodes),
            'node_sigmoid': tg.NodeSigmoid(),

            # Use this to force global to depend on previous global features
            'global': tg.GlobalLinear(features.Output.Global.LENGTH, global_features=mp_out_globals),
            'global_sigmoid': tg.GlobalSigmoid()

            # Use this if we want global = w * mean(nodes) + b
            # 'global': tg.GlobalLinear(features.Output.Global.LENGTH, node_features=1, aggregation='mean'),
        }))

    def forward(self, decoys: DecoyBatch):
        decoys = self.preprocessing(decoys)
        decoys = self.encoder(decoys)
        decoys = self.layers(decoys)
        decoys = self.readout(decoys)

        return decoys.evolve(
            node_features=decoys.node_features,
            num_edges_by_graph=None,
            edge_index_by_graph=None,
            edge_features=None,
            global_features=decoys.global_features,
            senders=None,
            receivers=None
        )


class ProteinGNNoGlobal(nn.Module):
    def __init__(
            self,
            min_dist=0,
            max_dist=20,
            rbf_size=16,
            residue_emb_size=32,
            separation_enc=True,
            enc_in_nodes=83,
            enc_in_edges=8,
            layers=1,
            mp_in_edges=8,
            mp_in_nodes=16,
            mp_out_edges=64,
            mp_out_nodes=128,
            dropout=0,
            batch_norm=False,
    ):
        super().__init__()
        self.layers = []

        self.preprocessing = nn.Sequential(OrderedDict({
            'rbf_distances': RbfDistanceEncodingEdges(min_dist, max_dist, rbf_size),
            'separation_encoding': SeparationEncodingEdges() if separation_enc else nn.Identity(),
            'residue_embedding': ResidueEmbedding(residue_emb_size),
            'duplicate_edges': DuplicateEdges(),
        }))

        # Edge feature shape: E -> mp_in_edges//2 -> mp_in_edges
        # Node feature shape: N -> mp_in_nodes//2 -> mp_in_nodes
        self.encoder = nn.Sequential(OrderedDict({
            'edge1': tg.EdgeLinear(out_features=mp_in_edges//2, edge_features=enc_in_edges),
            'edge1_dropout': tg.EdgeDropout(p=dropout) if dropout > 0 else nn.Identity(),
            'edge1_bn': tg.EdgeBatchNorm(num_features=mp_in_edges//2) if batch_norm else nn.Identity(),
            'edge1_relu': tg.EdgeReLU(),
            'edge2': tg.EdgeLinear(out_features=mp_in_edges, edge_features=mp_in_edges//2),
            'edge2_dropout': tg.EdgeDropout(p=dropout),
            'edge2_bn': tg.EdgeBatchNorm(num_features=mp_in_edges) if batch_norm else nn.Identity(),
            'edge2_relu': tg.EdgeReLU(),

            'node1': tg.NodeLinear(out_features=mp_in_nodes//2, node_features=enc_in_nodes),
            'node1_dropout': tg.NodeDropout(p=dropout) if dropout > 0 else nn.Identity(),
            'node1_bn': tg.NodeBatchNorm(num_features=mp_in_nodes//2) if batch_norm else nn.Identity(),
            'node1_relu': tg.NodeReLU(),
            'node2': tg.NodeLinear(out_features=mp_in_nodes, node_features=mp_in_nodes//2),
            'node2_dropout': tg.NodeDropout(p=dropout) if dropout > 0 else nn.Identity(),
            'node2_bn': tg.NodeBatchNorm(num_features=mp_in_nodes) if batch_norm else nn.Identity(),
            'node2_relu': tg.NodeReLU(),
        }))

        # Message passing layers: edge, node and global shapes go
        # from (mp_in_edges , mp_in_nodes , mp_in_globals )
        # to   (mp_out_edges, mp_out_nodes, mp_out_globals)
        # following powers of 2 in the number of steps given with the `layers` parameter (e.g. 10).
        mp_layers_output_sizes = np.power(2, np.linspace(
            np.log2((mp_in_edges, mp_in_nodes)),
            np.log2((mp_out_edges, mp_out_nodes)),
            num=layers, endpoint=False
        ).round().astype(int)).tolist()[1:] + [(mp_out_edges, mp_out_nodes)]

        in_e, in_n = mp_in_edges, mp_in_nodes
        for out_e, out_n in mp_layers_output_sizes:
            layer = nn.Sequential(OrderedDict({
                'edge1': tg.EdgeLinear(
                    out_features=out_e,
                    edge_features=in_e,
                    sender_features=in_n,
                ),
                'edge1_dropout': tg.EdgeDropout(p=dropout) if dropout > 0 else nn.Identity(),
                'edge1_bn': tg.EdgeBatchNorm(num_features=out_e) if batch_norm else nn.Identity(),
                'edge1_relu': tg.EdgeReLU(),

                'node1': tg.NodeLinear(
                    out_features=out_n,
                    node_features=in_n,
                    incoming_features=out_e,
                    aggregation='mean',
                ),
                'node1_dropout': tg.NodeDropout(p=dropout) if dropout > 0 else nn.Identity(),
                'node1_bn': tg.NodeBatchNorm(num_features=out_n) if batch_norm else nn.Identity(),
                'node1_relu': tg.NodeReLU(),
            }))
            self.layers.append(layer)
            in_e, in_n = out_e, out_n

        self.layers = torch.nn.Sequential(OrderedDict({f'layer_{i}': l for i, l in enumerate(self.layers)}))

        # Node feature shape: mp_out_nodes -> 1
        # Global feature shape: mp_out_nodes -> 1
        self.readout = nn.Sequential(OrderedDict({
            'global': tg.GlobalLinear(features.Output.Global.LENGTH, node_features=mp_out_nodes, aggregation='mean'),
            'global_sigmoid': tg.GlobalSigmoid(),

            'node': tg.NodeLinear(features.Output.Node.LENGTH, node_features=mp_out_nodes),
            'node_sigmoid': tg.NodeSigmoid(),
        }))

    def forward(self, decoys: DecoyBatch):
        decoys = self.preprocessing(decoys)
        decoys = self.encoder(decoys)
        decoys = self.layers(decoys)
        decoys = self.readout(decoys)

        return decoys.evolve(
            node_features=decoys.node_features,
            num_edges_by_graph=None,
            edge_index_by_graph=None,
            edge_features=None,
            global_features=decoys.global_features,
            senders=None,
            receivers=None
        )


class ResidueEmbedding(nn.Embedding):
    def __init__(self, residue_emb_size: int):
        super().__init__(num_embeddings=22, embedding_dim=residue_emb_size)

    def forward(self, decoys: DecoyBatch) -> DecoyBatch:
        residue_embeddings = super(ResidueEmbedding, self).forward(decoys.residues)
        return decoys.evolve(
            residues=None,
            node_features=torch.cat((
                residue_embeddings,
                decoys.node_features
            ), dim=1)
        )


class ScalarDistanceEncodingEdges(nn.Module):
    def forward(self, decoys: DecoyBatch):
        # Distances are encoded as simple scalars
        decoys = decoys.evolve(
            distances=None,
            edge_features=torch.cat((
                decoys.distances[:, None],
                decoys.edge_features
            ), dim=1),
        )

        return decoys


class RbfDistanceEncodingEdges(nn.Module):
    def __init__(self, min_dist: float, max_dist: float, size: int):
        super().__init__()
        if not 0 <= min_dist < max_dist:
            raise ValueError(f'Invalid RBF centers: 0 <= {min_dist} < {max_dist} is False')
        if size < 0:
            raise ValueError(f'Invalid RBF size: 0 < {size} is False')
        self.register_buffer('rbf_centers', torch.linspace(min_dist, max_dist, steps=size))

    def forward(self, decoys: DecoyBatch):
        # Distances are encoded using a equally spaced RBF kernels with unit variance
        distances_rbf = torch.exp(- (decoys.distances[:, None] - self.rbf_centers[None, :]) ** 2)

        decoys = decoys.evolve(
            distances=None,
            edge_features=torch.cat((
                distances_rbf,
                decoys.edge_features
            ), dim=1),
        )

        return decoys


class ScalarSeparationEncodingEdges(nn.Module):
    def forward(self, decoys: DecoyBatch):
        separation = decoys.receivers - decoys.senders - 1
        decoys = decoys.evolve(
            edge_features=torch.cat((
                decoys.edge_features,
                separation[:, None].float()
            ), dim=1),
        )
        return decoys


class CategoricalSeparationEncodingEdges(nn.Module):
    # In numpy we would do:
    # separation = [0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11]
    # bins = [0, 1, 2, 5, 10]
    # np.searchsorted(bins, separation, side='right') - 1
    # > [0, 1, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4]
    #
    # But torchsearchsorted does not have side=right. To get the same result:
    # (len(bins) - 1) - np.searchsorted(-bins[::-1], -separation, side='left')
    #
    # Also torchsearchsorted requires 2D float inputs and returns float outputs
    def __init__(self):
        super().__init__()
        # self.register_buffer('bins', torch.tensor([0, 1, 2, 3, 4, 5, 10]))
        self.register_buffer('bins', - torch.tensor([0, 1, 2, 3, 4, 5, 10]).flip(0).float().unsqueeze_(0))

    def forward(self, decoys: DecoyBatch):
        # separation = decoys.receivers - graphs.senders - 1
        # separation_cls = searchsorted(separation, self.bins, side='right') - 1

        separation = (decoys.senders - decoys.receivers + 1).float().unsqueeze_(0)
        separation_cls = (self.bins.numel() - 1) - searchsorted(self.bins, separation).squeeze_(0).long()

        separation_onehot = torch.zeros(decoys.num_edges, self.bins.numel(), device=decoys.senders.device)
        separation_onehot.scatter_(value=1., index=separation_cls.unsqueeze_(1), dim=1)

        decoys = decoys.evolve(
            edge_features=torch.cat((
                decoys.edge_features,
                separation_onehot
            ), dim=1),
        )

        return decoys


class DuplicateEdges(nn.Module):
    def __call__(self, decoys: DecoyBatch):
        decoys = decoys.evolve(
            senders=torch.cat((decoys.senders, decoys.receivers), dim=0),
            receivers=torch.cat((decoys.receivers, decoys.senders), dim=0),
            edge_features=decoys.edge_features.repeat(2, 1),
            num_edges_by_graph=decoys.num_edges_by_graph * 2,
            edge_index_by_graph=decoys.edge_index_by_graph.repeat(2)
        )
        return decoys