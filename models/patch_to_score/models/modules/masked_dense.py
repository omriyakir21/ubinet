import tensorflow as tf
import keras


class MaskedDense(keras.layers.Dense):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.supports_masking = True
        
    def _zero_out_masked(self, tensor, mask) -> tf.Tensor:
        if mask is not None:
            tensor = tensor * tf.expand_dims(tf.cast(mask, tensor.dtype), -1)
        return tensor

    def call(self, inputs, mask=None):
        inputs = self._zero_out_masked(inputs, mask)
        outputs = super().call(inputs)
        outputs = self._zero_out_masked(outputs, mask)
        return outputs
