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
        
    def _split_to_heads(self, tensor: tf.Tensor) -> tf.Tensor:
        batch, num_patches, attention_dimension = tensor.shape
        res = tf.reshape(tensor, (batch, num_patches, self.num_heads, self.head_dimension))
        res = tf.transpose(res, perm=[0, 2, 1, 3])  # (batch, num_heads, num_patches, head_dim)
        return res
    
    def _concat_heads(self, tensor: tf.Tensor) -> tf.Tensor:
        batch, num_heads, num_patches, head_dimension = tensor.shape
        res = tf.transpose(tensor, perm=[0, 2, 1, 3])
        res = tf.reshape(res, (batch, num_patches, self.attention_dimension))
        return res
    
    def call(self, inputs, training=False, mask=None):
        F = inputs[0]
        D = inputs[1]
        
        B = self.pairs_layernorm(D)
        B = tf.keras.layers.ReLU()(B)
        B = self.dense_pairs_heads(B)
        B = tf.transpose(B, perm=[0, 3, 1, 2])

        F = self.features_layernorm(F)
        Q = self.Wq(F)
        K = self.Wk(F)
        V = self.Wv(F)
        G = self.dense_gating(F)

        Q_reshaped = self._split_to_heads(Q)
        K_reshaped = self._split_to_heads(K)
        V_reshaped = self._split_to_heads(V)
        G_reshaped = self._split_to_heads(G)
        
        G_reshaped = tf.nn.sigmoid(G_reshaped)

        attention_scores = tf.einsum('bhpd,bhqd->bhpq', Q_reshaped, K_reshaped)
        attention_scores /= tf.sqrt(tf.cast(self.head_dimension, tf.float32))
        attention_scores += B

        if mask is not None:
            features_mask = mask[0]
            if features_mask is not None:
                # mask shape is (B, num_patches)
                # attention_scores shape is (B, num_heads, num_patches, num_patches)
                features_mask = tf.expand_dims(features_mask, axis=1)
                features_mask = tf.expand_dims(features_mask, axis=2)
                # Set attention scores to -inf where features_mask is False
                # This will effectively mask out those patches in the attention mechanism
                attention_scores = tf.where(features_mask, attention_scores, -1e9 * tf.ones_like(attention_scores))

        attention_weights = tf.nn.softmax(attention_scores, axis=-1)
        attention_output = tf.einsum('bhpq,bhqd->bhpd', attention_weights, V_reshaped)
        attention_output = tf.math.multiply(G_reshaped, attention_output)
        attention_output = self._concat_heads(attention_output)
        
        O = self.dense_output(attention_output)
        return O

    def compute_mask(self, inputs, mask=None):
        # Just return the input mask unchanged
        return mask
