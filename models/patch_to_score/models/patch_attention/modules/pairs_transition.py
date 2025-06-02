import uuid
import tensorflow as tf
from tensorflow.keras import layers


class PairsTransition(tf.keras.layers.Layer):
    def __init__(self, pairs_channel_dimension: int, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True  # Important! The pathces are masked
        
        self.pairs_channel_dimension = pairs_channel_dimension
        self.dense = tf.keras.layers.Dense(pairs_channel_dimension, use_bias=True, name='dense_pairs')
    
    def call(self, inputs, training=False, mask=None):
        x = tf.expand_dims(inputs, axis=-1)
        x = self.dense(x)
        return x

    def compute_mask(self, inputs, mask=None):
        # Just return the input mask unchanged
        # TODO: should expand last dim of the mask?
        return mask
