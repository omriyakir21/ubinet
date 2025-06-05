from typing import Tuple, List
import tensorflow as tf
from models.patch_to_score.models.modules import PairsTransition, TransformerEncoderMLP, PatchAttentionWithoutPairBias, GlobalSumPooling


def broadcast_shape(x, max_number_of_patches: int) -> tf.Tensor:
    x_expanded = tf.expand_dims(x, axis=1)
    x_broadcasted = tf.broadcast_to(
        x_expanded, [tf.shape(x)[0], max_number_of_patches, tf.shape(x)[-1]])
    return x_broadcasted


def create_inputs(input_shape: Tuple[int, int], max_number_of_patches: int):
    input_data = tf.keras.Input(shape=input_shape, name='features_input')
    size_value = tf.keras.Input(shape=(1,), name='number_of_residues_input')
    n_patches_hot_encoded_value = tf.keras.Input(
        shape=(max_number_of_patches + 1,), name='number_of_patches_input')
    return input_data, size_value, n_patches_hot_encoded_value


def create_broadcasted_features(n_patches_hot_encoded_value: tf.Tensor, max_number_of_patches: int, size_value: tf.Tensor, input_data: tf.Tensor):
    n_patches = tf.argmax(n_patches_hot_encoded_value, axis=1)[..., None]
    n_patches = tf.cast(n_patches, tf.float32)

    n_patches_broadcased = broadcast_shape(n_patches, max_number_of_patches)
    size_broadcased = broadcast_shape(size_value, max_number_of_patches)

    concat_input_data = tf.keras.layers.Concatenate(
        axis=-1)([input_data, n_patches_broadcased, size_broadcased])

    return concat_input_data


def mask_inputs(features: tf.Tensor):
    features = tf.keras.layers.Masking(mask_value=0.0)(features)
    return features


def apply_mlps(inputs: tf.Tensor, hidden_sizes: List[Tuple[int, int]], dropout_rate: float, activation: str) -> tf.Tensor:
    current_output = inputs
    for hidden_size in hidden_sizes:
        mlp = TransformerEncoderMLP(
            hidden_units=hidden_size, dropout_rate=dropout_rate, activation=activation)
        current_output = mlp(current_output)
        current_output = tf.keras.layers.ReLU()(
            current_output)
    return current_output


def create_masked_inputs(input_data: tf.Tensor, size_value: tf.Tensor, n_patches_hot_encoded_value: tf.Tensor, max_number_of_patches: int) -> Tuple[tf.Tensor, tf.Tensor]:
    mask_condition = tf.reduce_all(~tf.equal(input_data, 0), axis=-1)
    mask_condition = tf.cast(
        tf.reduce_any(input_data != 0, axis=-1), tf.float32)

    features = create_broadcasted_features(
        n_patches_hot_encoded_value, max_number_of_patches, size_value, input_data)

    # zero out broadcased features where mask is 0
    features = features * mask_condition[..., None]

    features, mask_inputs(features)
    return features


def build_model(features_mlp_hidden_sizes: List[Tuple[int, int]], features_mlp_dropout_rate: float,
                output_mlp_hidden_sizes: List[Tuple[int, int]], output_mlp_dropout_rate: float,
                attention_mlp_hidden_sizes: List[Tuple[int, int]], attention_mlp_dropout_rate: float,
                activation: str,
                input_shape: Tuple[int, int],
                max_number_of_patches: int,
                attention_dimension: int,
                num_heads: int) -> tf.keras.models.Model:
    '''
    :features_mlp_hidden_sizes: size of the hidden layers in the features MLP
    :features_mlp_dropout_rate: dropout_rate for the features MLP
    :output_mlp_hidden_sizes: size of the hidden layers in the output MLP
    :output_mlp_dropout_rate: dropout_rate for the output MLP
    :attention_mlp_hidden_sizes: size of the hidden layers in the attention MLP
    :attention_mlp_dropout_rate: dropout_rate for the attention MLP
    :activation: activation function to use in the MLPs
    :param input_shape: shape of the input data (number of patches, number of features) - usually (10, 9)
    :param max_number_of_patches: maximum number of patches
    :param attention_dimension: dimension of the patch attention layer
    :return: a Keras model
    '''
    input_data, size_value, n_patches_hot_encoded_value = create_inputs(
        input_shape, max_number_of_patches)
    features = create_masked_inputs(
        input_data, size_value, n_patches_hot_encoded_value, max_number_of_patches)
    F = apply_mlps(features, features_mlp_hidden_sizes,
                   features_mlp_dropout_rate, activation)
    attention_output = PatchAttentionWithoutPairBias(
        attention_dimension, num_heads)(F)
    attention_mlp_output = apply_mlps(
        attention_output, attention_mlp_hidden_sizes, attention_mlp_dropout_rate, activation)
    global_pooling_output = GlobalSumPooling(
        data_format='channels_last')(attention_mlp_output)
    output_mlp_output = apply_mlps(
        global_pooling_output, output_mlp_hidden_sizes, output_mlp_dropout_rate, activation)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(output_mlp_output)
    model = tf.keras.Model(inputs=[
                           input_data, size_value, n_patches_hot_encoded_value], outputs=output)
    return model
