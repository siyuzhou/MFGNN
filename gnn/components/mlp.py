from tensorflow import keras


class MLP(keras.layers.Layer):
    def __init__(self, units, dropout=0., batch_norm=False, kernel_l2=0., name=None):
        if not units:
            raise ValueError("'units' must not be empty")

        super().__init__(name=name)
        self.hidden_layers = []
        self.dropout_layers = []

        for i, unit in enumerate(units[:-1]):
            name = f'hidden{i}'
            layer = keras.layers.Dense(unit, activation='relu',
                                       kernel_regularizer=keras.regularizers.l2(kernel_l2),
                                       name=name)
            self.hidden_layers.append(name)
            setattr(self, name, layer)

            dropout_name = f'dropout{i}'
            dropout_layer = keras.layers.Dropout(dropout)
            self.dropout_layers.append(dropout_name)
            setattr(self, dropout_name, dropout_layer)

        self.out_layer = keras.layers.Dense(units[-1], activation='relu', name='out_layer')

        if batch_norm:
            self.batch_norm = keras.layers.BatchNormalization()
        else:
            self.batch_norm = None

    def call(self, x, training=False):
        for name, dropout_name in zip(self.hidden_layers, self.dropout_layers):
            layer = getattr(self, name)
            dropout_layer = getattr(self, dropout_name)

            x = layer(x)
            x = dropout_layer(x, training=training)

        x = self.out_layer(x)
        if self.batch_norm:
            return self.batch_norm(x, training=training)
        return x
