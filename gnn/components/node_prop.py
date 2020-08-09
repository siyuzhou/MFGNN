import tensorflow as tf


class NodePropagator(tf.keras.layers.Layer):
    """
    Pass message between every pair of nodes.
    """

    def call(self, node_states):
        # node_states shape [batch, num_nodes, out_units].
        num_nodes = node_states.shape[1]

        msg_from_source = tf.repeat(tf.expand_dims(node_states, 2), num_nodes, axis=2)
        msg_from_target = tf.repeat(tf.expand_dims(node_states, 1), num_nodes, axis=1)

        # msg_from_source and msg_from_target in shape [batch, num_nodes, num_nodes, out_units]
        node_msgs = tf.concat([msg_from_source, msg_from_target], axis=-1)

        return node_msgs


class NodePropagatorSparse(tf.keras.layers.Layer):
    def __init__(self, edges):
        if len(edges.shape) != 2 or edges.shape[0] != edges.shape[1]:
            raise ValueError('`edges` must be a square matrix')
        super().__init__()

        # Construct full connection matrix, mark source node and target node for each connection.
        # `self._edge_sources` and `self._edge_targets` with size [num_edges, num_nodes]
        edge_sources, edge_targets = tf.transpose(tf.where(edges))
        self._edge_sources = tf.one_hot(edge_sources, len(edges))
        self._edge_targets = tf.one_hot(edge_targets, len(edges))

    def call(self, node_states):
        # node_states shape [batch, num_nodes, out_units].
        msg_from_source = tf.transpose(tf.tensordot(node_states, self._edge_sources, axes=[[1], [1]]),
                                       perm=[0, 2, 1])
        msg_from_target = tf.transpose(tf.tensordot(node_states, self._edge_targets, axes=[[1], [1]]),
                                       perm=[0, 2, 1])
        # msg_from_source and msg_from_target in shape [batch, num_edges, out_units]
        node_msgs = tf.concat([msg_from_source, msg_from_target], axis=-1)

        return node_msgs
