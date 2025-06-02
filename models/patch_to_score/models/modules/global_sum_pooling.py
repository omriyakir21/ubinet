import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras.layers import GlobalAveragePooling1D


@tf.keras.utils.register_keras_serializable(package="MyCustomLayersEncoderMLP")
class GlobalSumPooling(GlobalAveragePooling1D):
    def __init__(self, data_format='channels_last', keepdims=False, **kwargs):
        super(GlobalSumPooling, self).__init__(
            data_format=data_format, **kwargs)
        self.keepdims = keepdims

    def call(self, inputs, mask=None):
        steps_axis = 1 if self.data_format == "channels_last" else 2
        if mask is not None:
            mask = tf.cast(mask, inputs[0].dtype)
            mask = tf.expand_dims(
                mask, 2 if self.data_format == "channels_last" else 1
            )
            inputs *= mask
            return backend.sum(
                inputs, axis=steps_axis, keepdims=self.keepdims
            )

    def get_config(self):
        config = super(GlobalSumPooling, self).get_config()
        config.update({
            "data_format": self.data_format,
            "keepdims": self.keepdims
        })
        return config
