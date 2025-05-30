from typing import Tuple, List
import tensorflow as tf
from models.patch_to_score.models.pts_encoder_mlps.utils import GlobalSumPooling
from models.patch_to_score.models.pts_encoder_mlps.transformer_encoder_mlp import TransformerEncoderMLP
from models.patch_to_score.models.patch_attention_v2.patch_attention_with_pair_bias import PatchAttentionWithPairBias


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


def mask_inputs(features: tf.Tensor, coordinates: tf.Tensor):
    features = tf.keras.layers.Masking(mask_value=0.0)(features)
    coordinates = tf.keras.layers.Masking(
        mask_value=0.0)(coordinates)
    return features, coordinates


def apply_mlps(current_output: tf.Tensor, hidden_sizes: List[Tuple[int, int]], dropout_rate: float, activation: str) -> tf.Tensor:
    for hidden_size in hidden_sizes:
        mlp = TransformerEncoderMLP(
            hidden_units=hidden_size, dropout_rate=dropout_rate, activation=activation)
        current_output = mlp(current_output)
        current_output = tf.keras.layers.ReLU()(
            current_output)
    return current_output


def build_model(output_mlp_hidden_sizes: List[Tuple[int, int]], output_mlp_dropout_rate: float,
                attention_mlp_hidden_sizes: List[Tuple[int, int]], attention_mlp_dropout_rate: float,
                activation: str,
                input_shape: Tuple[int, int],
                max_number_of_patches: int,
                attention_dimension: int,
                num_heads: int) -> tf.keras.models.Model:
    '''
    :param m_a: size of the hidden layers in the MLP of the components
    :param m_c: size of the hidden layers in the MLP of the concatenated global sum output and size + n_patches MLP output
    :param n_layers: number of layers in each of the MLPs
    :param input_shape: shape of the input data (number of patches, number of features)
    :param max_number_of_patches: maximum number of patches
    :return: a Keras model
    '''
    input_data, coordinates, size_value, n_patches_hot_encoded_value = create_inputs(
        input_shape, max_number_of_patches)
    features = create_broadcasted_features(
        n_patches_hot_encoded_value, max_number_of_patches, size_value, input_data)
    features, coordinates = mask_inputs(features, coordinates)

    pairwise_distances = tf.norm(
        tf.expand_dims(coordinates, axis=1) - tf.expand_dims(coordinates, axis=2), axis=-1)
    
    feature_transition = tf.keras.layers.Dense(attention_dimension, use_bias=True, activation=activation)
    pairs_transition = tf.keras.layers.Dense(attention_dimension, use_bias=True, activation=activation)
    
    F = feature_transition(features)
    D = tf.expand_dims(pairwise_distances, axis=-1)
    D = pairs_transition(D)
    
    current_output = PatchAttentionWithPairBias(
        attention_dimension, num_heads)([F, D])

    current_output = apply_mlps(
        current_output, attention_mlp_hidden_sizes, attention_mlp_dropout_rate, activation)

    global_pooling_output = GlobalSumPooling(
        data_format='channels_last')(current_output)
    current_output = global_pooling_output

    current_output = apply_mlps(
        current_output, output_mlp_hidden_sizes, output_mlp_dropout_rate, activation)

    before_sigmoid_output = current_output

    output = tf.keras.layers.Dense(
        1, activation='sigmoid')(before_sigmoid_output)
    model = tf.keras.Model(
        inputs=[input_data, coordinates, size_value, n_patches_hot_encoded_value], outputs=output)
    return model
