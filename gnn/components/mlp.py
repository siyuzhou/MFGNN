from tensorflow import keras


class MLP(keras.layers.Layer):
    def __init__(self, units, dropout=0., batch_norm=False, kernel_l2=0., activation='relu', name=None):
        if not units:
            raise ValueError("'units' must not be empty")

        super().__init__(name=name)
        self.hidden_layers = []
        self.dropout_layers = []

        if units:
            for i, unit in enumerate(units[:-1]):
                layer = keras.layers.Dense(unit, activation='relu',
                                           kernel_regularizer=keras.regularizers.l2(kernel_l2),
                                           name=f'hidden{i}')
                self.hidden_layers.append(layer)

                dropout_layer = keras.layers.Dropout(dropout, name=f'dropout{i}')
                self.dropout_layers.append(dropout_layer)

            self.out_layer = keras.layers.Dense(units[-1], activation=activation, name='out_layer')
        else:
            self.hidden_layers.append(keras.layers.Lambda(lambda x: x))
            self.dropout_layers.append(keras.layers.Dropout(dropout))

        if batch_norm:
            self.batch_norm = keras.layers.BatchNormalization()
        else:
            self.batch_norm = None

    def call(self, x, training=False):
        for layer, dropout in zip(self.hidden_layers, self.dropout_layers):
            x = layer(x)
            x = dropout(x, training=training)

        x = self.out_layer(x)
        if self.batch_norm:
            return self.batch_norm(x, training=training)
        return x
