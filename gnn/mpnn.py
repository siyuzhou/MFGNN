import torch
import torch.nn as nn
import numpy as np

from .modules import GraphConv, Conv1D

class MPNN(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.pred_steps = params['pred_steps']
        self._time_seg_len = params['time_seg_len']

        if self._time_seg_len > 1:
            self.conv1d = Conv1D(params['ndims'], params['cnn']['filters'])
        else:
            self.conv1d = nn.Identity()

        self.graph_conv = GraphConv(params['num_nodes'], params['edge_type'], params)
        self.dense = nn.Linear(params['node_decoder']['hidden_units'][-1], params['ndims'])

    def _pred_next(self, time_segs, edges):

        condensed_state = self.conv1d(time_segs.transpose(1, 2)).transpose(1, 2)
        # condensed_state shape [bath, 1, num_nodes, filters]

        node_state = self.graph_conv(condensed_state, edges)

        # Predicted difference added to the prev state.
        prev_state = time_segs[:, -1:, :, :]
        next_state = prev_state + self.dense(node_state)
        return next_state

    def forward(self, time_segs, edges):
        # time_segs shape [batch, time_seg_len, num_nodes, ndims]
        # edges shape [batch, num_nodes, num_nodes, edge_types], one-hot label along last axis.
        extended_time_segs = time_segs

        for i in range(self.pred_steps):
            next_state = self._pred_next(extended_time_segs[:, i:, :, :], edges)
            extended_time_segs = torch.cat([extended_time_segs, next_state], 1)

        return extended_time_segs[:, self._time_seg_len:, :, :]
