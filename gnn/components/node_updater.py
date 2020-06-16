import tensorflow as tf
from .mlp import MLP


class NodeUpdater(tf.keras.layers.Layer):
    def __init__(self, ndim, params, activation=None, name=None):
        super().__init__(name=name)

        self.updater = MLP(params['hidden_units'],
                           params['dropout'],
                           params['batch_norm'],
                           params['kernel_l2'])

        self.out_layer = tf.keras.layers.Dense(ndim, activation=activation)

    def call(self, node_states, messages, training=False):
        inputs = tf.concat([node_states, messages], axis=-1)

        new_node_states = self.out_layer(self.updater(inputs, training=training))
        return new_node_states


class NodeLinearUpdater(tf.keras.layers.Layer):
    def __init__(self, ndim, bias=False, a=1., b=1., name=None):
        super().__init__(name=name)

        self.a = a
        self.b = b

        self.weights = tf.keras.layers.Dense(ndim, use_bias=bias)

    def call(self, node_states, messages, training=False):
        new_node_states = self.a * node_states + self.b * self.weights(messages)
        return new_node_states
