import uuid
import tensorflow as tf
from tensorflow.keras import layers


class PatchAttentionWithPairBias(tf.keras.layers.Layer):
    def __init__(self, pairs_dimension: int, attention_dimension: int, num_heads: int, **kwargs):
        """
        Args:
            pairs_dimension: Hidden dimension for pairs transition
            attention_dimension: Dimension for all attention operations
            num_heads: Attention heads, TODO: currently only supports single head (num_heads=1)
        """
        super().__init__(**kwargs)
        self.supports_masking = True  # Important! The pathces are masked
                
        self.pairs_dimension = pairs_dimension
        self.attention_dimension = attention_dimension
        
        self.Wk = layers.Dense(attention_dimension, use_bias=False, name=f'Wk_{uuid.uuid4()}')
        self.Wq = layers.Dense(attention_dimension, use_bias=False, name=f'Wq_{uuid.uuid4()}')
        self.Wv = layers.Dense(attention_dimension, use_bias=False, name=f'Wv_{uuid.uuid4()}')
        
        self.dense_pairs_transition = layers.Dense(pairs_dimension, use_bias=True, name=f'dense_pairs_{uuid.uuid4()}')
        self.dense_pairs_heads = layers.Dense(num_heads, use_bias=False, name=f'dense_pairs_{uuid.uuid4()}')
        self.dense_output = layers.Dense(attention_dimension, use_bias=True, name=f'dense_output_{uuid.uuid4()}')
        
        self.pairs_layernorm = layers.LayerNormalization()
        self.features_layernorm = layers.LayerNormalization()

    def build(self, input_shape):
        super().build(input_shape)
    
    def call(self, inputs, training=False, mask=None):
        F = inputs[0]
        D = inputs[1]
        
        F = self.features_layernorm(F)
        
        # TODO: multihead
        Q = self.Wq(F)
        K = self.Wk(F)
        V = self.Wv(F)

        # TODO: multihead
        B = self.pairs_layernorm(D)        
        B = self.dense_pairs(B)
        B = self.dense_pairs_heads(B)
        
        scalar = (1 / tf.sqrt(self.attention_dimension))
        A = tf.matmul(Q, K, transpose_b=True)
        A += B
        A = tf.nn.softmax(scalar * A, axis=-1)
        
        O = tf.matmul(A, V)
        O = tf.math.reduce_sum(O, axis=-1)
        
        # TODO: multihead (concat & project out)  
        
        return O  # Mask will be automatically passed if supports_masking=True

    def compute_mask(self, inputs, mask=None):
        # Just return the input mask unchanged
        return mask
