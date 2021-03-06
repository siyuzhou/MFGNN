import tensorflow as tf
from tensorflow import keras

from ..convolutions import Conv1D, GraphConv


class MPNN(keras.Model):
    def __init__(self, params):
        super().__init__(name='MPNN')

        # NOTE: For the moment assume Conv1D is always applied
        self.pred_steps = params['pred_steps']
        self.time_seg_len = params['time_seg_len']

        if self.time_seg_len > 1:
            self.conv1d = Conv1D(params['cnn']['filters'], name='Conv1D')
        else:
            self.conv1d = keras.layers.Lambda(lambda x: x, name='NoConv')

        num_graph_convs = params.get('num_graph_convs', 1)
        self.graph_convs = []
        for i in range(num_graph_convs):
            self.graph_convs.append(GraphConv(params['num_nodes'],
                                              params['edge_type'],
                                              params, name=f'GraphConv_{i}'))

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
        node_state = tf.squeeze(condensed_state, axis=2)
        # condensed_state shape [batch, num_nodes, filters]
        for conv in self.graph_convs:
            node_state = conv(node_state, edges, training)

        # Predicted difference added to the prev state.
        # The last state in each timeseries of the stack.
        prev_state = time_segs[:, :, -1, :]
        next_state = prev_state + node_state
        return next_state

    def call(self, inputs, training=False):
        # time_segs shape [batch, time_seg_len, num_nodes, ndims]
        # edges shape [batch, num_nodes, num_nodes, edge_types], one-hot label along last axis.
        time_segs, edges = inputs

        extended_time_segs = tf.transpose(time_segs, [0, 2, 1, 3])

        for i in range(self.pred_steps):
            next_state = self._pred_next(extended_time_segs[:, :, i:, :], edges,
                                         training=training)
            next_state = tf.expand_dims(next_state, axis=2)
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
