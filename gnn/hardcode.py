import numpy as np
import tensorflow as tf
from tensorflow import keras

from .modules import *


class EdgeAggregatorAVG(keras.layers.Layer):
    """
    Average messages from incoming edges to the node.
    """
    def __init__(self, edges):
        super().__init__()

        # `edge_sources` and `edge_targets` in shape [num_edges, num_nodes].
        edge_targets = np.where(edges)[1]
        self._edge_targets = tf.one_hot(edge_targets, len(edges))

    def call(self, edge_msg, edge_types):
        # edge_msg shape [batch, num_edges, edge_type, out_units]
        edge_msg_sum = tf.transpose(tf.tensordot(edge_msg, self._edge_targets, axes=[[1], [0]]),
                                    perm=[0, 3, 1, 2])  # Shape [batch, num_nodes, edge_type, out_units].
        
        in_degree_by_type = tf.reduce_sum(edge_types, axis=1) # Shape [batch, num_nodes, edge_type]
        
        edge_msg_avg_by_type = edge_msg_sum / tf.expand_dims(in_degree_by_type, axis=-1)
        # Add messsages of all edge types.
        edge_msg_sum = tf.reduce_sum(edge_msg_avg_by_type, axis=2, keepdims=True)
        return edge_msg_sum


class GraphAVGConv(keras.layers.Layer):
    def __init__(self, graph_size, edge_type, params):
        super().__init__(name='GraphConv')

        fc = np.ones((graph_size, graph_size))
        self.node_prop = NodePropagator(fc)
        self.edge_aggr = EdgeAggregatorAVG(fc)

        self.edge_encoder = EdgeEncoder(edge_type, params['edge_encoder'])

        self.node_decoder = MLP(params['node_decoder']['hidden_units'],
                                params['node_decoder']['dropout'],
                                params['node_decoder']['batch_norm'],
                                params['node_decoder']['kernel_l2'],
                                name='node_decoder')

    def call(self, node_states, edges, training=False):
        # Propagate node states.
        node_msgs = self.node_prop(node_states)

        # Form edges. Shape [batch, num_edges, edge_type, units]
        edge_msgs = self.edge_encoder(node_msgs, edges, training)

        # Edge aggregation. Shape [batch, num_nodes, 1, units]
        edge_msgs_aggr = self.edge_aggr(edge_msgs, edges)

        # Update node_states
        node_states = self.node_decoder(tf.concat([node_states, edge_msgs_aggr], axis=-1), training=training)

        return node_states



class HardCodedMFGNN(keras.Model):
    def __init__(self, params):
        super().__init__(name='HardCodedMFGNN')

        # NOTE: For the moment assume Conv1D is always applied
        self.pred_steps = params['pred_steps']
        self.time_seg_len = params['time_seg_len']

        if self.time_seg_len > 1:
            self.conv1d = Conv1D(params['cnn']['filters'], name='Conv1D')
        else:
            self.conv1d = keras.layers.Lambda(lambda x: x)

        self.gc_filters = [GraphConv(params['num_nodes'], params['edge_type'], params, name='GraphConv')]
        self.gc_filters.append(GraphAVGConv(params['num_nodes'], params['edge_type'], params, name='GraphAVGConv'))
        
        self.node_decoder = MLP(params['node_decoder']['hidden_units'],
                                params['node_decoder']['dropout'],
                                params['node_decoder']['batch_norm'],
                                params['node_decoder']['kernel_l2'],
                                name='node_decoder')

        self.dense = keras.layers.Dense(params['ndims'], name='out_layer')

    def build(self, input_shape):
        t = keras.layers.Input(input_shape[0][1:])
        e = keras.layers.Input(input_shape[1][1:])

        inputs = [t, e]

        self.call(inputs)
        self.built = True
        return inputs

    def _pred_next(self, time_segs, edges, training=False):
        condensed_state = self.conv1d(time_segs)
        # condensed_state shape [batch, num_nodes, 1, filters]

        node_msgs = []
        for graph_conv in self.gc_filters:
            node_msg = graph_conv(condensed_state, edges, training)
            node_msgs.append(node_msg)

        node_msg_sum = tf.reduce_sum(tf.stack(node_msgs, axis=0), axis=0)
        node_state = self.node_decoder(tf.concat([condensed_state, node_msg_sum], axis=-1))
        # Predicted difference added to the prev state.
        # The last state in each timeseries of the stack.
        prev_state = time_segs[:, :, -1:, :]
        next_state = prev_state + self.dense(node_state)
        return next_state

    def call(self, inputs, training=False):
        # time_segs shape [batch, time_seg_len, num_nodes, ndims]
        # edges shape [batch, num_nodes, num_nodes, edge_types], one-hot label along last axis.
        time_segs, edges = inputs

        # Transposed to [batch, num_nodes, time_seg_len, ndims].
        extended_time_segs = tf.transpose(time_segs, [0, 2, 1, 3])

        for i in range(self.pred_steps):
            next_state = self._pred_next(
                extended_time_segs[:, :, i:, :], edges, training=training)
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

        input_shape = [(None, params['time_seg_len'], params['num_nodes'], params['ndims']),
                       (None, params['num_nodes'], params['num_nodes'], params['edge_type']+1)]

        inputs = model.build(input_shape)

        if return_inputs:
            return model, inputs

        return model