from typing import Tuple
import tensorflow as tf
from models.patch_to_score.models.pts_without_mlp_b.utils import GlobalSumPooling


def broadcast_shape(x, max_number_of_patches: int) -> tf.Tensor:
    x_expanded = tf.expand_dims(x, axis=1)
    x_broadcasted = tf.broadcast_to(x_expanded, [tf.shape(x)[0], max_number_of_patches, tf.shape(x)[-1]])
    return x_broadcasted


def build_model(m_a: int, m_c: int, n_layers: int, input_shape: Tuple[int, int],
                max_number_of_patches: int) -> tf.keras.models.Model:
    '''
    :param m_a: size of the hidden layers in the MLP of the components
    :param m_c: size of the hidden layers in the MLP of the concatenated global sum output and size + n_patches MLP output
    :param n_layers: number of layers in each of the MLPs
    :param input_shape: shape of the input data (number of patches, number of features)
    :param max_number_of_patches: maximum number of patches
    :return: a Keras model
    '''
    input_data = tf.keras.Input(shape=input_shape, name='patches_input')
    size_value = tf.keras.Input(shape=(1,), name='extra_value_input')
    n_patches_hot_encoded_value = tf.keras.Input(
        shape=(max_number_of_patches + 1,), name='hot_encoded_value_input')
    n_patches = tf.argmax(n_patches_hot_encoded_value, axis=1)[..., None]
    n_patches = tf.cast(n_patches, tf.float32)
    
    n_patches_broadcased = broadcast_shape(n_patches, max_number_of_patches)
    size_broadcased = broadcast_shape(size_value, max_number_of_patches)
    
    concat_input_data = tf.keras.layers.Concatenate(axis=-1)([input_data, n_patches_broadcased, size_broadcased])
    masked_input = tf.keras.layers.Masking(mask_value=0.0)(concat_input_data)

    currentOutput = masked_input
    for i in range(n_layers):
        dense_output = tf.keras.layers.Dense(
            m_a, activation='linear')(currentOutput)
        batchNorm = tf.keras.layers.BatchNormalization(
            momentum=0.75)(dense_output)
        activation = tf.keras.layers.ReLU()(batchNorm)
        currentOutput = activation

    global_pooling_output = GlobalSumPooling(
        data_format='channels_last')(currentOutput)

    currentOutput = global_pooling_output
    for i in range(n_layers):
        dense_output = tf.keras.layers.Dense(
            m_c, activation='linear')(currentOutput)
        batchNorm = tf.keras.layers.BatchNormalization(
            momentum=0.75)(dense_output)
        activation = tf.keras.layers.ReLU()(batchNorm)
        currentOutput = activation

    before_sigmoid_output = currentOutput

    output = tf.keras.layers.Dense(
        1, activation='sigmoid')(before_sigmoid_output)
    model = tf.keras.Model(
        inputs=[input_data, size_value, n_patches_hot_encoded_value], outputs=output)
    return model
