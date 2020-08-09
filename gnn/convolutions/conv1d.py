from tensorflow import keras


class Conv1D(keras.layers.Layer):
    """
    Condense and abstract the time segments.
    """

    def __init__(self, filters, name=None):
        if not filters:
            raise ValueError("'filters' must not be empty")

        super().__init__(name=name)
        # time segment length before being reduced to 1 by Conv1D
        # self.seg_len = 2 * len(filters) + 1

        self.conv1d_layers = []
        for channels in filters:
            conv = keras.layers.TimeDistributed(
                keras.layers.Conv1D(channels, 3, activation='relu', name=name))
            self.conv1d_layers.append(conv)

    def call(self, time_segs):
        # Node state encoder with 1D convolution along timesteps and across ndims as channels.
        condensed_state = time_segs
        for conv in self.conv1d_layers:
            condensed_state = conv(condensed_state)

        return condensed_state
