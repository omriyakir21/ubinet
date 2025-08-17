import uuid
import tensorflow as tf
from tensorflow.keras import layers


class TransformerEncoderMLP(tf.keras.layers.Layer):
    def __init__(self, hidden_units, dropout_rate, activation, **kwargs):
        """
        Args:
            hidden_units: Tuple/List of two integers for the two dense layers.
            dropout_rate: Dropout rate for both dropout layers.
            activation: Activation function for the first dense layer.
        """
        super().__init__(**kwargs)
        self.supports_masking = True  # Important! The pathces are masked
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
        
        # Used if input and output dims don't match
        self.skip_proj = None

    def build(self, input_shape):
        input_dim = input_shape[-1]
        output_dim = self.hidden_units[-1]

        # Create projection layer only if dimensions mismatch
        if input_dim != output_dim:
            print(f'initializing projection for skip connection: {input_dim} -> {output_dim}')
            random_id = str(uuid.uuid4().hex[:10])
            self.skip_proj = layers.Dense(output_dim, name=f'skip_proj_{random_id}')  # patch for tensorflow name overlapping issue

        super().build(input_shape)
    
    def call(self, inputs, training=False, mask=None):
        x = self.dense1(inputs)
        x = self.activation_layer(x)
        x = self.dropout1(x, training=training)

        x = self.dense2(x)
        x = self.dropout2(x, training=training)
        
        # Skip connection
        residual = inputs if self.skip_proj is None else self.skip_proj(inputs)
        x = x + residual
        
        x = self.layernorm(x)
        return x

    def compute_mask(self, inputs, mask=None):
        # Just return the input mask unchanged
        return mask
