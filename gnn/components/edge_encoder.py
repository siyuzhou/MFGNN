import tensorflow as tf
from .mlp import MLP


class EdgeEncoder(tf.keras.layers.Layer):
    """
    Propagate messages to edge from the two nodes connected via edge encoders.
    """

    def __init__(self, edge_type, encoder_params):
        super().__init__()

        self.edge_type = edge_type

        self.edge_encoders = [MLP(encoder_params['hidden_units'],
                                  encoder_params['dropout'],
                                  encoder_params['batch_norm'],
                                  encoder_params['kernel_l2'],
                                  name=f'edge_encoder_{i}')
                              for i in range(1, self.edge_type+1)]

    def call(self, node_msgs, edges, training=False):
        # `node_msgs` shape [batch, num_nodes*num_nodes, units]
        # `edges` shape [batch, num_nodes, num_nodes, num_edge_label]
        edge_types = tf.expand_dims(edges, axis=-1)
        # Shape [batch, num_nodes, num_nodes, num_edge_label, 1]
        # edge_types = tf.reshape(edges, [-1, num_nodes*num_nodes, num_edge_label, 1])

        encoded_msgs_by_type = []
        for i in range(self.edge_type):
            # mlp_encoder for each edge type.
            encoded_msgs = self.edge_encoders[i](node_msgs, training=training)

            encoded_msgs_by_type.append(encoded_msgs)

        encoded_msgs_by_type = tf.stack(encoded_msgs_by_type, axis=3)
        # Shape [batch, num_nodes, num_nodes, edge_types, units]

        # Only encoded message of the type same as the edge type gets retaind.
        # Force skip 0 type, 0 means no connection, no message.
        edge_msgs = tf.multiply(encoded_msgs_by_type, edge_types[:, :, :, 1:, :])

        return edge_msgs
