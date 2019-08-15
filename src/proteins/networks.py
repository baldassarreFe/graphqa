from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np

import torchgraphs as tg

from . import features


class ProteinGN(nn.Module):
    def __init__(
            self,
            layers=1,
            hidden_size_in_edges=8,
            hidden_size_in_nodes=16,
            hidden_size_in_globals=4,
            hidden_size_out_edges=64,
            hidden_size_out_nodes=128,
            hidden_size_out_globals=32,
            dropout=0,
            batch_norm=False,
    ):
        super().__init__()
        self.layers = []

        # Edge feature shape: 2 -> hidden_size_in_edges//2 -> hidden_size_in_edges
        # Node feature shape: 83 -> hidden_size_in_nodes -> hidden_size_in_nodes//2
        # Global feature shape: None -> hidden_size_in_globals
        self.encoder = nn.Sequential(OrderedDict({
            'edge1': tg.EdgeLinear(out_features=hidden_size_in_edges//2, edge_features=2),
            'edge1_dropout': tg.EdgeDropout(p=dropout) if dropout > 0 else nn.Identity(),
            'edge1_bn': tg.EdgeBatchNorm(num_features=hidden_size_in_edges//2) if batch_norm else nn.Identity(),
            'edge1_relu': tg.EdgeReLU(),
            'edge2': tg.EdgeLinear(out_features=hidden_size_in_edges, edge_features=hidden_size_in_edges//2),
            'edge2_dropout': tg.EdgeDropout(p=dropout),
            'edge2_bn': tg.EdgeBatchNorm(num_features=hidden_size_in_edges) if batch_norm else nn.Identity(),
            'edge2_relu': tg.EdgeReLU(),

            'node1': tg.NodeLinear(out_features=hidden_size_in_nodes//2, node_features=83),
            'node1_dropout': tg.NodeDropout(p=dropout) if dropout > 0 else nn.Identity(),
            'node1_bn': tg.NodeBatchNorm(num_features=hidden_size_in_nodes//2) if batch_norm else nn.Identity(),
            'node1_relu': tg.NodeReLU(),
            'node2': tg.NodeLinear(out_features=hidden_size_in_nodes, node_features=hidden_size_in_nodes//2),
            'node2_dropout': tg.NodeDropout(p=dropout) if dropout > 0 else nn.Identity(),
            'node2_bn': tg.NodeBatchNorm(num_features=hidden_size_in_nodes) if batch_norm else nn.Identity(),
            'node2_relu': tg.NodeReLU(),

            'global': tg.GlobalLinear(out_features=hidden_size_in_globals, bias=True),
            'global_dropout': tg.GlobalDropout(p=dropout) if dropout > 0 else nn.Identity(),
            'global_relu': tg.GlobalReLU(),
        }))

        # Edge, node and global shapes decrease from
        # (hidden_size_in_edges, hidden_size_in_nodes, hidden_size_in_globals)
        # to
        # (hidden_size_out_edges, hidden_size_out_nodes, hidden_size_out_globals)
        # following powers of 2 in the number of steps given as parameter (e.g. 10).
        hidden_layers_output_sizes = np.power(2, np.linspace(
            np.log2((hidden_size_in_edges, hidden_size_in_nodes, hidden_size_in_globals)),
            np.log2((hidden_size_out_edges, hidden_size_out_nodes, hidden_size_out_globals)),
            num=layers, endpoint=False
        ).round().astype(int)).tolist()[1:] + [(hidden_size_out_edges, hidden_size_out_nodes, hidden_size_out_globals)]

        in_e, in_n, in_g = hidden_size_in_edges, hidden_size_in_nodes, hidden_size_in_globals
        for out_e, out_n, out_g in hidden_layers_output_sizes:
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
            self.layers.append(layer)
            in_e, in_n, in_g = out_e, out_n, out_g

        self.layers = torch.nn.Sequential(OrderedDict({f'layer_{i}': l for i, l in enumerate(self.layers)}))

        # Node feature shape: hidden_size_out_nodes -> 1
        # Global feature shape: hidden_size_out_globals -> 1
        self.readout = nn.Sequential(OrderedDict({
            'node': tg.NodeLinear(features.Output.Node.LENGTH, node_features=hidden_size_out_nodes),
            'node_sigmoid': tg.NodeSigmoid(),
            'global': tg.GlobalLinear(features.Output.Global.LENGTH,
                                      global_features=hidden_size_out_globals, node_features=1, aggregation='mean'),
            'global_sigmoid': tg.GlobalSigmoid()
        }))

    def forward(self, graphs):
        graphs = self.encoder(graphs)
        graphs = self.layers(graphs)
        graphs = self.readout(graphs)

        return graphs.evolve(
            node_features=graphs.node_features,
            num_edges_by_graph=None,
            edge_index_by_graph=None,
            edge_features=None,
            global_features=graphs.global_features,
            senders=None,
            receivers=None
        )
