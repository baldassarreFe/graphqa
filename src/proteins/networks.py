import math
from collections import OrderedDict
from torch import nn

import torchgraphs as tg


class ProteinGN(nn.Module):
    def __init__(self, node_features, edge_features, aggregation='mean', hops=1):
        super().__init__()
        self.hops = hops

        # Edge feature shape: 1 -> 4 -> 8
        # Node feature shape: 83 -> 32 -> 16
        # Global feature shape: None -> 4
        self.encoder = nn.Sequential(
            nn.Sequential(OrderedDict({
                'edge': tg.EdgeLinear(out_features=4, edge_features=edge_features),
                'edge_relu': tg.EdgeReLU(),
                'node': tg.NodeLinear(out_features=32, node_features=node_features),
                'node_relu': tg.NodeReLU(),
            })),
            nn.Sequential(OrderedDict({
                'edge': tg.EdgeLinear(out_features=8, edge_features=4),
                'edge_relu': tg.EdgeReLU(),
                'node': tg.NodeLinear(out_features=16, node_features=32),
                'node_relu': tg.NodeReLU(),
                'global': tg.GlobalLinear(node_features=16, out_features=4, aggregation='mean'),
            }))
        )

        # Edge feature shape: 8 -> 8
        # Node feature shape: 16 -> 16
        # Global feature shape: 4 -> 4
        self.hidden = nn.Sequential(OrderedDict({
            'edge': tg.EdgeLinear(
                out_features=8, edge_features=8, sender_features=16, global_features=4),
            'edge_relu': tg.EdgeReLU(),
            'node': tg.NodeLinear(
                out_features=16, node_features=16, incoming_features=8, global_features=4, aggregation=aggregation),
            'node_relu': tg.NodeReLU(),
            'global': tg.GlobalLinear(
                out_features=4, edge_features=8, node_features=16, global_features=4, aggregation=aggregation),
            'global_relu': tg.GlobalReLU(),
        }))

        # Node feature shape: 16 -> 1
        # Global feature shape: 4 -> 1
        self.readout_nodes = nn.Sequential(OrderedDict({
            'node': tg.NodeLinear(1, node_features=16),
            'node_sigmoid': tg.NodeSigmoid()
        }))
        self.readout_globals = nn.Sequential(OrderedDict({
            'global': tg.GlobalLinear(1, global_features=4),
            'global_sigmoid': tg.GlobalSigmoid()
        }))

        _reset_parameters(self)

    def forward(self, graphs):
        graphs = self.encoder(graphs)

        for hop in range(self.hops):
            graphs = self.hidden(graphs)

        nodes = self.readout_nodes(graphs).node_features
        globals = self.readout_globals(graphs).global_features

        return graphs.evolve(
            node_features=nodes,
            num_edges=0,
            num_edges_by_graph=None,
            edge_features=None,
            global_features=globals,
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
