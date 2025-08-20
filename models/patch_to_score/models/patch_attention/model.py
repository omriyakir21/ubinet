from typing import Tuple, List
import tensorflow as tf
import numpy as np
from models.patch_to_score.models.modules import TransformerEncoderMLP, PatchAttentionWithPairBias, GlobalSumPooling, \
    GaussianKernel, initialize_gaussian_kernel_uniform


def broadcast_shape(x, max_number_of_patches: int) -> tf.Tensor:
    x_expanded = tf.expand_dims(x, axis=1)
    x_broadcasted = tf.broadcast_to(
        x_expanded, [tf.shape(x)[0], max_number_of_patches, tf.shape(x)[-1]])
    return x_broadcasted


def create_inputs(input_shape: Tuple[int, int], max_number_of_patches: int):
    input_data = tf.keras.Input(shape=input_shape, name='features_input')
    input_coordinates = tf.keras.Input(
        shape=(max_number_of_patches, 3), name='coordinates_input')
    size_value = tf.keras.Input(shape=(1,), name='number_of_residues_input')
    n_patches_hot_encoded_value = tf.keras.Input(
        shape=(max_number_of_patches + 1,), name='number_of_patches_input')
    return input_data, input_coordinates, size_value, n_patches_hot_encoded_value


def create_broadcasted_features(n_patches_hot_encoded_value: tf.Tensor, max_number_of_patches: int, size_value: tf.Tensor, input_data: tf.Tensor):
    n_patches = tf.argmax(n_patches_hot_encoded_value, axis=1)[..., None]
    n_patches = tf.cast(n_patches, tf.float32)

    n_patches_broadcased = broadcast_shape(n_patches, max_number_of_patches)
    size_broadcased = broadcast_shape(size_value, max_number_of_patches)

    concat_input_data = tf.keras.layers.Concatenate(
        axis=-1)([input_data, n_patches_broadcased, size_broadcased])

    return concat_input_data


def apply_mlps(inputs: tf.Tensor, hidden_sizes: List[Tuple[int, int]], dropout_rate: float, activation: str) -> tf.Tensor:
    current_output = inputs
    for hidden_size in hidden_sizes:
        mlp = TransformerEncoderMLP(
            hidden_units=hidden_size, dropout_rate=dropout_rate, activation=activation)
        current_output = mlp(current_output)
        current_output = tf.keras.layers.ReLU()(
            current_output)
    return current_output


def create_masked_inputs(input_data: tf.Tensor, coordinates: tf.Tensor, size_value: tf.Tensor, n_patches_hot_encoded_value: tf.Tensor, max_number_of_patches: int) -> Tuple[tf.Tensor, tf.Tensor]:
    mask_condition = tf.reduce_all(~tf.equal(input_data, 0), axis=-1)
    mask_condition = tf.cast(
        tf.reduce_any(input_data != 0, axis=-1), tf.float32)

    features = create_broadcasted_features(
        n_patches_hot_encoded_value, max_number_of_patches, size_value, input_data)
    pairwise_distances = tf.norm(
        tf.expand_dims(coordinates, axis=1) - tf.expand_dims(coordinates, axis=2), axis=-1)
    pairwise_distances = tf.cast(pairwise_distances, tf.float32)

    # zero out broadcased features where mask is 0
    features = features * mask_condition[..., None]

    mask_matrix = mask_condition[:, :, None] * mask_condition[:, None, :]
    # zero out pairwise distances where mask is 0
    pairwise_distances = pairwise_distances * mask_matrix
    mask_matrix -= 1
    # change masked values to -1 to avoid masking the diagonal
    pairwise_distances = pairwise_distances + mask_matrix
    pairwise_distances = tf.expand_dims(
        pairwise_distances, axis=-1)  # add channel dimension

    features = tf.keras.layers.Masking(mask_value=0.0)(features)
    pairwise_distances = tf.keras.layers.Masking(
        mask_value=-1.0)(pairwise_distances)

    return features, pairwise_distances


def patch_to_score(features: tf.Tensor, pairwise_distances: tf.Tensor,
                   features_mlp_hidden_sizes: List[Tuple[int, int]], features_mlp_dropout_rate: float,
                   output_mlp_hidden_sizes: List[Tuple[int, int]], output_mlp_dropout_rate: float,
                   attention_mlp_hidden_sizes: List[Tuple[int, int]], attention_mlp_dropout_rate: float,
                   activation: str,
                   attention_dimension: int,
                   pairs_channel_dimension: int,
                   gaussian_xrange: Tuple[float, float],
                   num_heads: int,
                   use_pair_bias: bool) -> tf.Tensor:

    F = apply_mlps(features, features_mlp_hidden_sizes,
                   features_mlp_dropout_rate, activation)

    if use_pair_bias:
        kernel = initialize_gaussian_kernel_uniform(
            gaussian_xrange, pairs_channel_dimension)
        D = kernel(pairwise_distances)
        patch_attention_input = [F, D]
    else:
        patch_attention_input = [F]

    attention_output = PatchAttentionWithPairBias(
        attention_dimension, num_heads, use_pair_bias)(patch_attention_input)

    attention_output = tf.keras.layers.Add()(
        [F, attention_output])  # residual connection

    attention_mlp_output = apply_mlps(
        attention_output, attention_mlp_hidden_sizes, attention_mlp_dropout_rate, activation)

    global_pooling_output = GlobalSumPooling(
        data_format='channels_last')(attention_mlp_output)

    # from this point on, since we reduced the patch demension, there's no more masking (GlobalSumPooling didn't pass on the mask)
    output_mlp_output = apply_mlps(
        global_pooling_output, output_mlp_hidden_sizes, output_mlp_dropout_rate, activation)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(output_mlp_output)
    return output


def build_model(features_mlp_hidden_sizes: List[Tuple[int, int]], features_mlp_dropout_rate: float,
                output_mlp_hidden_sizes: List[Tuple[int, int]], output_mlp_dropout_rate: float,
                attention_mlp_hidden_sizes: List[Tuple[int, int]], attention_mlp_dropout_rate: float,
                activation: str,
                input_shape: Tuple[int, int],
                max_number_of_patches: int,
                attention_dimension: int,
                pairs_channel_dimension: int,
                gaussian_xrange: Tuple[float, float],
                num_heads: int,
                use_pair_bias: bool) -> tf.keras.models.Model:
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
    :param pairs_channel_dimension: dimension of the pairs transition layer
    :param gaussian_xrange: range of the gaussian kernel, will be initialized uniformly in range, with pairs_channel_dimension centers
    :param num_heads: number of heads for multi-head attention
    :param use_pair_bias: should use pair bias in the patch attention module 
    :return: a Keras model
    '''
    input_data, coordinates, size_value, n_patches_hot_encoded_value = create_inputs(
        input_shape, max_number_of_patches)
    features, pairwise_distances = create_masked_inputs(
        input_data, coordinates, size_value, n_patches_hot_encoded_value, max_number_of_patches)

    output = patch_to_score(features, pairwise_distances,
                            features_mlp_hidden_sizes, features_mlp_dropout_rate,
                            output_mlp_hidden_sizes, output_mlp_dropout_rate,
                            attention_mlp_hidden_sizes, attention_mlp_dropout_rate,
                            activation,
                            attention_dimension,
                            pairs_channel_dimension,
                            gaussian_xrange,
                            num_heads,
                            use_pair_bias)

    model = tf.keras.Model(inputs=[
                           input_data, coordinates, size_value, n_patches_hot_encoded_value], outputs=output)
    return model
