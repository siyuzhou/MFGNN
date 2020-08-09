import tensorflow as tf
from tensorflow import keras


class EdgeSumAggregator(keras.layers.Layer):
    """
    Sum up messages from incoming edges to the node.
    """

    def call(self, edge_msgs, node_states, edges):
        # edge_msg shape [batch, num_nodes, num_nodes, edge_type, out_units]

        # Sum messsages of all edge types. Shape becomes [batch, num_nodes, out_units]
        edge_msg_sum = tf.reduce_sum(edge_msgs, axis=[1, 3])

        return edge_msg_sum


class EdgeSumAggregatorSparse(keras.layers.Layer):
    def __init__(self, edges):
        if len(edges.shape) != 2 or edges.shape[0] != edges.shape[1]:
            raise ValueError('`edges` must be a square matrix')
        super().__init__()
        # `edge_targets` in shape [num_edges, num_nodes].
        edge_targets = tf.where(edges)[:, 1]
        self._edge_targets = tf.one_hot(edge_targets, len(edges))

    def call(self, edge_msgs):
        # edge_msg shape [batch, num_edges, edge_type, out_units]
        # Sum edge msgs in each neighborhood.
        edge_msg_sum = tf.transpose(tf.tensordot(edge_msg_sum, self._edge_targets, axes=[[1], [0]]),
                                    perm=[0, 2, 1])  # Shape [batch, num_nodes, out_units].

        return edge_msg_sum


class EdgeMeanAggregator(keras.layers.Layer):
    """
    Average messages from incoming edges to the node.
    """

    def call(self, edge_msgs, node_states, edges):
        # edge_msg shape [batch, num_nodes, num_nodes, edge_type, out_units]
        # edges shape [batch, num_nodes, num_nodes, num_edge_labels]
        in_degrees = tf.reduce_sum(edges[:, :, :, 1:], axis=1, keepdims=True)
        denoms = tf.expand_dims(in_degrees, -1)
        denoms = tf.where(denoms == 0, tf.ones_like(denoms), denoms)

        edge_msg_mean = tf.reduce_sum(edge_msgs / denoms, axis=[1, 3])
        return edge_msg_mean


class EdgeAttentionAggregator(keras.layers.Layer):
    """
    Weighted sum of messages from incoming edges. Weights are 
    determined by attention.
    """

    def __init__(self, nheads=1):
        super().__init__()

        self.attentions = []
        for a in range(nheads):
            self.attentions.append(keras.layers.Dense(1, activation='elu'))

        self.softmax = keras.layers.Softmax(axis=1)

    def call(self, edge_msgs, node_states, edges):
        # `edge_msg` shape [batch, num_nodes, num_nodes, edge_type, out_units]
        # `node_states` shape [batch, num_nodes, out_units].
        # `edges` shape [batch, num_nodes, num_nodes, num_edge_label]
        num_nodes, edge_type = edge_msgs.shape[2:4]

        node_states = tf.expand_dims(tf.expand_dims(node_states, 1), 3)
        # New shape [batch, 1, num_nodes, 1, out_units]
        node_states = tf.tile(node_states, [1, num_nodes, 1, edge_type, 1])

        multihead_edge_msgs = []
        for a in self.attentions:
            attentions = a(tf.concat([edge_msgs, node_states], axis=-1))
            # `attentions` shape [batch, num_nodes, num_nodes, edge_type, 1]

            bias = (1 - tf.expand_dims(edges[:, :, :, 1:], -1)) * -1e6
            weights = self.softmax(attentions + bias)

            weighted_edge_msgs = weights * edge_msgs

            edge_msg_sum = tf.reduce_sum(weighted_edge_msgs, axis=[1, 3])
            # shape [batch, num_nodes, out_units]
            multihead_edge_msgs.append(edge_msg_sum)

        # shape [batch, num_nodes, out_units * nheads]
        return tf.concat(multihead_edge_msgs, axis=-1)
