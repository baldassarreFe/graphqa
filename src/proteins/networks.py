import math
from collections import OrderedDict
import torch
from torch import nn

import torchgraphs as tg


class ProteinGN(nn.Module):
    def __init__(
            self,
            layers=1,
            size_edges=2,
            size_nodes=4,
            size_globals=1,
            dropout=0
    ):
        super().__init__()
        self.layers = []

        # Edge feature shape: 2 -> 4 -> 16
        # Node feature shape: 83 -> 64 -> 32
        # Global feature shape: None -> 8
        self.encoder = nn.Sequential(OrderedDict({
            'edge1': tg.EdgeLinear(out_features=4, edge_features=2),
            'edge1_relu': tg.EdgeReLU(),
            'edge1_dropout': tg.EdgeDropout(p=dropout),
            'edge2': tg.EdgeLinear(out_features=16, edge_features=4),
            'edge2_relu': tg.EdgeReLU(),
            'edge2_dropout': tg.EdgeDropout(p=dropout),

            'node1': tg.NodeLinear(out_features=64, node_features=83),
            'node1_relu': tg.NodeReLU(),
            'node1_dropout': tg.NodeDropout(p=dropout),
            'node2': tg.NodeLinear(out_features=32, node_features=64),
            'node2_relu': tg.NodeReLU(),
            'node2_dropout': tg.NodeDropout(p=dropout),

            'global': tg.GlobalLinear(out_features=8, bias=True),
            'global_relu': tg.GlobalReLU(),
            'global_dropout': tg.GlobalDropout(p=dropout),
        }))

        # Edge, node and global shapes linearly decrease from (16, 32, 8)
        # to the sizes given as parameters (e.g. 2, 4, 1)
        # in the number of steps given as parameter (e.g. 10)
        hidden_size_edges = torch.linspace(16, size_edges, layers+1).int().tolist()
        hidden_size_nodes = torch.linspace(32, size_nodes, layers+1).int().tolist()
        hidden_size_globals= torch.linspace(8, size_globals, layers+1).int().tolist()

        for in_e, in_n, in_g, out_e, out_n, out_g in zip(
            hidden_size_edges, hidden_size_nodes, hidden_size_globals,
            hidden_size_edges[1:], hidden_size_nodes[1:], hidden_size_globals[1:]
        ):
            layer = nn.Sequential(OrderedDict({
                'edge': tg.EdgeLinear(
                    out_features=out_e,
                    edge_features=in_e,
                    sender_features=in_n,
                    global_features=in_g
                ),
                'edge_relu': tg.EdgeReLU(),
                'edge_dropout': tg.EdgeDropout(p=dropout),

                'node': tg.NodeLinear(
                    out_features=out_n,
                    node_features=in_n,
                    incoming_features=out_e,
                    global_features=in_g,
                    aggregation='mean',
                ),
                'node_relu': tg.NodeReLU(),
                'node_dropout': tg.NodeDropout(p=dropout),

                'global': tg.GlobalLinear(
                    out_features=out_g,
                    edge_features=out_e,
                    node_features=out_n,
                    global_features=in_g,
                    aggregation='mean',
                ),
                'global_relu': tg.GlobalReLU(),
                'global_dropout': tg.GlobalDropout(p=dropout),
            }))
            self.layers.append(layer)

        self.layers = torch.nn.Sequential(OrderedDict({f'layer_{i}': l for i, l in enumerate(self.layers)}))

        # Node feature shape: size_nodes -> 1
        # Global feature shape: size_globals -> 1
        self.readout = nn.Sequential(OrderedDict({
            'node': tg.NodeLinear(1, node_features=size_nodes),
            'node_sigmoid': tg.NodeSigmoid(),
            'global': tg.GlobalLinear(1, global_features=size_globals, node_features=1, aggregation='mean'),
            'global_sigmoid': tg.GlobalSigmoid()
        }))

        _reset_parameters(self)

    def forward(self, graphs):
        graphs = self.encoder(graphs)
        graphs = self.layers(graphs)
        graphs = self.readout(graphs)

        return graphs.evolve(
            node_features=graphs.node_features,
            num_edges=0,
            num_edges_by_graph=None,
            edge_features=None,
            global_features=graphs.global_features,
            senders=None,
            receivers=None
        )


def _reset_parameters(module):
    for name, param in module.named_parameters():
        if 'bias' in name:
            bound = 1 / math.sqrt(param.numel())
            nn.init.uniform_(param, -bound, bound)
        else:
            nn.init.kaiming_uniform_(param, a=math.sqrt(5))
