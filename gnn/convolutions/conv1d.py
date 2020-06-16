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
        self.seg_len = 2 * len(filters) + 1

        self.conv1d_layers = []
        for i, channels in enumerate(filters):
            name = f'conv{i}'
            layer = keras.layers.TimeDistributed(
                keras.layers.Conv1D(channels, 3, activation='relu', name=name))
            self.conv1d_layers.append(name)
            setattr(self, name, layer)

        self.channels = channels

    def call(self, time_segs):
        # Node state encoder with 1D convolution along timesteps and across ndims as channels.
        encoded_state = time_segs
        for name in self.conv1d_layers:
            conv = getattr(self, name)
            encoded_state = conv(encoded_state)

        return encoded_state
