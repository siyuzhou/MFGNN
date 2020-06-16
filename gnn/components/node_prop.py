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
