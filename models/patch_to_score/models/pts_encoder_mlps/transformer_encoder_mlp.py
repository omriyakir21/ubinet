
import tensorflow as tf
from tensorflow.keras import layers


class TransformerEncoderMLP(tf.keras.layers.Layer):
    def __init__(self, hidden_units, dropout_rate=0.1, activation='relu', **kwargs):
        """
        Args:
            hidden_units: Tuple/List of two integers for the two dense layers.
            dropout_rate: Dropout rate for both dropout layers.
            activation: Activation function for the first dense layer.
        """
        super().__init__(**kwargs)
        assert len(hidden_units) == 2, "Provide two units: (first_dense_units, second_dense_units)"
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.activation = activation

        self.dense1 = layers.Dense(hidden_units[0])
        self.activation_layer = layers.Activation(activation)
        self.dropout1 = layers.Dropout(dropout_rate)

        self.dense2 = layers.Dense(hidden_units[1])
        self.dropout2 = layers.Dropout(dropout_rate)

        self.layernorm = layers.LayerNormalization()

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.activation_layer(x)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.dropout2(x, training=training)
        x = x + inputs
        x = self.layernorm(x)
        return x
