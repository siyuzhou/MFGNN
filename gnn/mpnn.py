import numpy as np
import tensorflow as tf
from tensorflow import keras

from .modules import *
from .utils import fc_matrix


class GraphConv(keras.Model):
    def __init__(self, params):
        super().__init__(name='GraphConv')

        # Whether edge type used for model.
        self.edge_type = params['edge_type']

        self.edge_encoders = [MLP(params['edge_encoder']['hidden_units'],
                                    params['edge_encoder']['dropout'],
                                    params['edge_encoder']['batch_norm'],
                                    params['edge_encoder']['kernel_l2'],
                                    name=f'edge_encoder_{i}')
                                for i in range(1, self.edge_type+1)]

        self.node_decoder = MLP(params['node_decoder']['hidden_units'],
                                params['node_decoder']['dropout'],
                                params['node_decoder']['batch_norm'],
                                params['node_decoder']['kernel_l2'],
                                name='node_decoder')

        edges = fc_matrix(params['nagents'])
        self.node_aggr = NodeAggregator(edges)
        self.edge_aggr = EdgeAggregator(edges)

    def call(self, inputs, training=False):
        # Form edges. Shape [batch, num_edges, 1, filters]
        node_states, edge_types = inputs
        edge_msg = self.node_aggr(node_states)

    
        encoded_msg_by_type = []
        for i in range(self.edge_type):
            # mlp_encoder for each edge type.
            encoded_msg = self.edge_encoders[i](edge_msg, training=training)

            encoded_msg_by_type.append(encoded_msg)

        encoded_msg_by_type = tf.concat(encoded_msg_by_type, axis=2)
        # Shape [batch, num_edges, edge_types, hidden_units]

        edge_msg = tf.reduce_sum(tf.multiply(encoded_msg_by_type,
                                                edge_types[:, :, 1:, :]), 
                                                # `1:` skip 0 type, no connection, no message.
                                    axis=2,
                                    keepdims=True)


        # Edge aggregation. Shape [batch, num_nodes, 1, filters]
        node_msg = self.edge_aggr(edge_msg)

        node_state = tf.concat([node_states, node_msg], axis=-1)
        node_state = self.node_decoder(node_state, training=training)

        return node_state


class MPNN(keras.Model):
    def __init__(self, params):
        super().__init__(name='SwarmNet')

        # NOTE: For the moment assume Conv1D is always applied
        self.pred_steps = params['pred_steps']
        self.time_seg_len = params['time_seg_len']

        if self.time_seg_len > 1:
            self.conv1d = Conv1D(params['cnn']['filters'], name='Conv1D')
        else:
            self.conv1d = keras.layers.Lambda(lambda x: x)

        self.graph_conv = GraphConv(params)
        self.dense = keras.layers.Dense(params['ndims'], name='out_layer')

    def build(self, input_shape):
        t = keras.layers.Input(input_shape[0][1:])
        e = keras.layers.Input(input_shape[1][1:])
        inputs = [t, e]

        self.call(inputs)
        self.built = True
        return inputs

    def _pred_next(self, time_segs, edge_types, training=False):
        condensed_state = self.conv1d(time_segs)
        # condensed_state shape [batch, num_agents, 1, filters]

        node_state = self.graph_conv([condensed_state, edge_types], training)

        # Predicted difference added to the prev state.
        # The last state in each timeseries of the stack.
        prev_state = time_segs[:, :, -1:, :]
        next_state = prev_state + self.dense(node_state)
        return next_state

    def call(self, inputs, training=False):
        # time_segs shape [batch, time_seg_len, num_agents, ndims]
        # Transpose to [batch, num_agents, time_seg_len,ndims]
        time_segs = inputs[0]
        edge_types = tf.expand_dims(inputs[1], axis=3)
        # Shape [None, n_edges, n_types, 1]

        extended_time_segs = tf.transpose(time_segs, [0, 2, 1, 3])

        for i in range(self.pred_steps):
            next_state = self._pred_next(
                extended_time_segs[:, :, i:, :], edge_types, training=training)
            extended_time_segs = tf.concat([extended_time_segs, next_state], axis=2)

        # Transpose back to [batch, time_seg_len+pred_steps, num_agetns, ndims]
        extended_time_segs = tf.transpose(extended_time_segs, [0, 2, 1, 3])

        # Return only the predicted part of extended_time_segs
        return extended_time_segs[:, self.time_seg_len:, :, :]

    @classmethod
    def build_model(cls, params, return_inputs=False):
        model = cls(params)

        optimizer = keras.optimizers.Adam(lr=params['learning_rate'])

        model.compile(optimizer, loss='mse')

        input_shape = [(None, params['time_seg_len'], params['nagents'], params['ndims']),
                       (None, params['nagents']*(params['nagents']-1), params['edge_type']+1)]
        
        inputs = model.build(input_shape)

        if return_inputs:
            return model, inputs

        return model
