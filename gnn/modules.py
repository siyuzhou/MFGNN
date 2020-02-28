import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .utils import one_hot


class MLP(nn.Module):
    def __init__(self, input_size, units, activation=None):
        if not units:
            raise ValueError("'units' must not be empty")

        super().__init__()
        self.hidden_layers = nn.ModuleList([nn.Linear(uin, uout)
                                            for uin, uout in zip([input_size]+units[:-1], units)])

    def forward(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
            x = F.relu(x)

        return x


class Conv1D(nn.Module):
    """
    Apply 1D convolution along the second last dimension.
    Input shape: (..., l, d)
    Output shape: (..., l', f)
    """
    def __init__(self, input_size, filters):
        if not filters:
            raise ValueError("'filters' must not be empty")
        super().__init__()

        self.seg_len = 2 * len(filters) + 1

        self.conv1ds = nn.ModuleList([nn.Conv1d(fin, fout, 3)
                                      for fin, fout in zip([3]+filters[:-1], filters)])
         
    def forward(self, x):
        shape = x.shape
        # Conv1d applies to input with shape (N, C, L) and outputs (N, C', L').
        # `x` is to be reshaped and transposed.
        x_3d = x.view(-1, shape[-2], shape[-1]).transpose(-1, -2)

        for conv in self.conv1ds:
            x_3d = conv(x_3d)
            x_3d = F.relu(x_3d)

        return x_3d.view(shape[:-2] + x_3d.shape[-2:]).transpose(-1, -2)
        

class NodePropagator(nn.Module):
    def __init__(self, edges):
        super().__init__()

        edge_sources, edge_targets = np.where(edges)
        self._edge_sources = torch.tensor(one_hot(edge_sources, len(edges)), dtype=torch.float32)
        self._edge_targets = torch.tensor(one_hot(edge_targets, len(edges)), dtype=torch.float32)

    def forward(self, node_states):
        # node_states shape [batch, 1, num_nodes, units]
        msg_from_source = torch.matmul(self._edge_sources, node_states)
        msg_from_target = torch.matmul(self._edge_targets, node_states)
        
        node_msgs = torch.cat([msg_from_source, msg_from_target], -1)
        # Shape [batch, 1, num_edges, 2*units]

        return node_msgs

    def _apply(self, fn):
        super()._apply(fn)
        self._edge_sources = fn(self._edge_sources)
        self._edge_targets = fn(self._edge_targets)

        return self


class EdgeAggregator(nn.Module):
    def __init__(self, edges):
        super().__init__()

        edge_targets = np.where(edges)[1]
        self._edge_targets = torch.tensor(one_hot(edge_targets, len(edges)), dtype=torch.float32)

    def forward(self, edge_msgs):
        # edgemsgs shape [batch, edge_type, num_edges, units]
        edge_msgs_sum = torch.matmul(self._edge_targets.t(), edge_msgs)
        # Add messages of all edge types.
        edge_msgs_sum = torch.sum(edge_msgs_sum, dim=1, keepdim=True)
        return edge_msgs_sum

    def _apply(self, fn):
        super()._apply(fn)
        self._edge_targets = fn(self._edge_targets)
        return self


class EdgeEncoder(nn.Module):
    def __init__(self, num_edge_types, encoder_params):
        super().__init__()

        self.edge_type = num_edge_types

        self.encoders = nn.ModuleList([MLP(encoder_params['input_size'],
                                           encoder_params['hidden_units'])
                                       for _ in range(1, self.edge_type+1)])

    def forward(self, node_msgs, edges):
        # `node_msgs` shape [batch, 1, num_edges, units]
        # `edges` shape [batch, num_nodes, num_nodes, num_edge_label]
        num_nodes, num_edge_label = edges.shape[2:]
        edge_types = edges.view(-1, num_nodes*num_nodes, num_edge_label, 1).transpose(1, 2)
        # Shape [batch, num_edge_label, num_edges, 1]

        encoded_msgs_by_type = [encoder(node_msgs)
                                for encoder in self.encoders]
        
        encoded_msgs_by_type = torch.cat(encoded_msgs_by_type, 1)
        # Shape [batch, edge_type, num_edges, units]

        # Only encoded message of the type same as the edge type gets retaind.
        # Force skip 0 type, 0 means no connection, no message.
        edge_msgs = encoded_msgs_by_type * edge_types[:, 1:, :, :]
        return edge_msgs
        

class GraphConv(nn.Module):
    def __init__(self, graph_size, edge_type, params):
        super().__init__()

        fc = np.ones((graph_size, graph_size))
        self.node_prop = NodePropagator(fc)
        self.edge_aggr = EdgeAggregator(fc)

        self.edge_encoder = EdgeEncoder(edge_type, params['edge_encoder'])

        self.node_decoder = MLP(params['node_decoder']['input_size'],
                                params['node_decoder']['hidden_units'])

    def forward(self, node_states, edges):
        node_msgs = self.node_prop(node_states)

        edge_msgs = self.edge_encoder(node_msgs, edges)

        edge_msgs_aggr = self.edge_aggr(edge_msgs)

        node_states = self.node_decoder(torch.cat([node_states, edge_msgs_aggr], -1))

        return node_states
        