import tensorflow as tf
from ..components import *


class GraphConv(tf.keras.layers.Layer):
    def __init__(self, graph_size, edge_type, params, name=None):
        super().__init__(name=name)

        self.node_prop = NodePropagator()

        if params['edge_aggregator']['type'] == "attention":
            self.edge_aggr = EdgeAttentionAggregator(
                params['edge_aggregator']['units'],
                params['edge_aggregator']['heads'])
        elif params['edge_aggregator']['type'] == "mean":
            self.edge_aggr = EdgeMeanAggregator()
            print('mean used')
        elif params['edge_aggregator']['type'] == 'sum':
            self.edge_aggr = EdgeSumAggregator()
        else:
            raise ValueError('edge_aggregator type unknown')

        self.edge_encoder = EdgeEncoder(edge_type, params['edge_encoder'])

        self.node_updater = NodeUpdater(params['node_updater'])

    def call(self, node_states, edges, training=False):
        # Propagate node states.
        node_msgs = self.node_prop(node_states)

        # Form edges. Shape [batch, num_edges, edge_type, units]
        edge_msgs = self.edge_encoder(node_msgs, edges, training)

        # Edge aggregation. Shape [batch, num_nodes, units]
        edge_msgs_aggr = self.edge_aggr(edge_msgs, node_states, edges)

        # Update node_states
        node_states = self.node_updater(node_states, edge_msgs_aggr, training)

        return node_states
