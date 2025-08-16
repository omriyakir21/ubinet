import tensorflow as tf
import keras


class MaskedDense(keras.layers.Dense):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.supports_masking = True

    def call(self, inputs, mask=None):
        if mask is not None:
            inputs = inputs * tf.expand_dims(tf.cast(mask, inputs.dtype), -1)
        outputs = super().call(inputs)
        return outputs

    def compute_mask(self, inputs, mask=None):
        return mask
