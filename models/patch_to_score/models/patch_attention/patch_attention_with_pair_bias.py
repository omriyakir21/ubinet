import uuid
import tensorflow as tf
from tensorflow.keras import layers


class PatchAttentionWithPairBias(tf.keras.layers.Layer):
    def __init__(self, attention_dimension: int, num_heads: int, **kwargs):
        """
        Args:
            attention_dimension: Dimension for all attention operations
            num_heads: Attention heads
        """
        super().__init__(**kwargs)
        
        # TODO: change to support D matrix with channel dimension
        
        assert attention_dimension % num_heads == 0, "Attention dimension must be divisible by number of heads."
        self.supports_masking = True  # Important! The pathces are masked        
        
        self.num_heads = num_heads
        self.attention_dimension = attention_dimension
        self.head_dimension = self.attention_dimension // self.num_heads
        
        self.Wk = layers.Dense(self.attention_dimension, use_bias=False, name=f'Wk_{uuid.uuid4()}')
        self.Wq = layers.Dense(self.attention_dimension, use_bias=False, name=f'Wq_{uuid.uuid4()}')
        self.Wv = layers.Dense(self.attention_dimension, use_bias=False, name=f'Wv_{uuid.uuid4()}')
        self.dense_gating = layers.Dense(self.attention_dimension, use_bias=True, name=f'dense_gating_{uuid.uuid4()}')

        # dense_pairs_transition = layers.Dense(pairs_dimension, use_bias=True, name=f'dense_pairs_{uuid.uuid4()}')
        self.dense_pairs_heads = layers.Dense(self.num_heads, use_bias=False, name=f'dense_pairs_{uuid.uuid4()}')
        self.dense_output = layers.Dense(self.attention_dimension, use_bias=True, name=f'dense_output_{uuid.uuid4()}')

        self.pairs_layernorm = layers.LayerNormalization()
        self.features_layernorm = layers.LayerNormalization()
    
    def build(self, input_shape):
        super().build(input_shape)
    
    def call(self, inputs, training=False, mask=None):
        F = inputs[0]
        D = inputs[1]
        
        B = self.pairs_layernorm(D)
        B = self.dense_pairs_heads(B)
        B = tf.reshape(B, (-1, B.shape[-1], B.shape[1], B.shape[2]))

        F = self.features_layernorm(F)
        Q = self.Wq(F)
        K = self.Wk(F)
        V = self.Wv(F)
        G = self.dense_gating(F)

        Q_reshaped = tf.reshape(Q, (-1, self.num_heads, Q.shape[1], self.head_dimension))
        K_reshaped = tf.reshape(K, (-1, self.num_heads, K.shape[1], self.head_dimension))
        V_reshaped = tf.reshape(V, (-1, self.num_heads, V.shape[1], self.head_dimension))
        G_reshaped = tf.reshape(G, (-1, self.num_heads, G.shape[1], self.head_dimension))
        G_reshaped = tf.nn.sigmoid(G_reshaped)

        attention_scores = tf.einsum('bhpd,bhqd->bhpq', Q_reshaped, K_reshaped)
        attention_scores /= tf.sqrt(tf.cast(self.head_dimension, tf.float32))
        attention_scores += B

        attention_weights = tf.nn.softmax(attention_scores, axis=-1)
        attention_output = tf.einsum('bhpp,bhpd->bhpd', attention_weights, V_reshaped)
        attention_output = tf.math.multiply(G_reshaped, attention_output)

        attention_output = tf.reshape(attention_output, (-1, attention_output.shape[2], self.attention_dimension))
        O = self.dense_output(attention_output)
        return O

    def compute_mask(self, inputs, mask=None):
        # Just return the input mask unchanged
        return mask
