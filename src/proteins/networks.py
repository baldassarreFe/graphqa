import math
from collections import OrderedDict
from torch import nn

import torchgraphs as tg


class ProteinGN(nn.Module):
    def __init__(self, node_features, edge_features, aggregation='mean', hops=1, hidden=4):
        super().__init__()
        self.hops = hops

        # Edge feature shape: 1 -> hidden -> 2*hidden
        # Node feature shape: 83 -> 8*hidden -> 4*hidden
        # Global feature shape: None -> hidden
        self.encoder = nn.Sequential(
            nn.Sequential(OrderedDict({
                'edge': tg.EdgeLinear(out_features=hidden, edge_features=edge_features),
                'edge_relu': tg.EdgeReLU(),
                'node': tg.NodeLinear(out_features=8 * hidden, node_features=node_features),
                'node_relu': tg.NodeReLU(),
            })),
            nn.Sequential(OrderedDict({
                'edge': tg.EdgeLinear(out_features=2 * hidden, edge_features=hidden),
                'edge_relu': tg.EdgeReLU(),
                'node': tg.NodeLinear(out_features=4 * hidden, node_features=8 * hidden),
                'node_relu': tg.NodeReLU(),
                'global': tg.GlobalLinear(out_features=hidden, node_features=4 * hidden, aggregation='mean'),
            }))
        )

        # Edge feature shape: 2*hidden -> 2*hidden
        # Node feature shape: 4*hidden -> 4*hidden
        # Global feature shape: hidden -> hidden
        self.hidden = nn.Sequential(OrderedDict({
            'edge': tg.EdgeLinear(out_features=2 * hidden,
                                  edge_features=2 * hidden, sender_features=4 * hidden, global_features=hidden),
            'edge_relu': tg.EdgeReLU(),
            'node': tg.NodeLinear(out_features=4 * hidden, aggregation=aggregation,
                                  node_features=4 * hidden, incoming_features=2 * hidden, global_features=hidden),
            'node_relu': tg.NodeReLU(),
            'global': tg.GlobalLinear(out_features=hidden, aggregation=aggregation,
                                      edge_features=2 * hidden, node_features=4 * hidden, global_features=hidden),
            'global_relu': tg.GlobalReLU(),
        }))

        # Node feature shape: 4*hidden -> 1
        # Global feature shape: hidden -> 1
        self.readout_nodes = nn.Sequential(OrderedDict({
            'node': tg.NodeLinear(1, node_features=4 * hidden),
            'node_sigmoid': tg.NodeSigmoid()
        }))
        self.readout_globals = nn.Sequential(OrderedDict({
            'global': tg.GlobalLinear(1, global_features=hidden),
            'global_sigmoid': tg.GlobalSigmoid()
        }))

        _reset_parameters(self)

    def forward(self, graphs):
        graphs = self.encoder(graphs)

        for hop in range(self.hops):
            residual = self.hidden(graphs)
            graphs.node_features = graphs.node_features + residual.node_features
            graphs.edge_features = graphs.edge_features + residual.edge_features
            graphs.global_features = graphs.global_features + residual.global_features

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
