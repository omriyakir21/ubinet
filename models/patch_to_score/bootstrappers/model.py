from typing import Tuple, List
from models.patch_to_score import utils
from models.patch_to_score.models.pts_without_mlp_b.model import build_model as build_pts_wihout_mlp_b_model
from models.patch_to_score.models.pts_encoder_mlps.model import build_model as build_pts_encoder_mlp_model
import tensorflow as tf


def bootstrap_patch_to_score_original(n_layers: int, m_a: int, m_b: int, m_c: int,
                                      max_number_of_patches: int, input_shape: Tuple[int, int]) -> tf.keras.models.Model:
    architecture_dict = {
        'm_a': m_a,
        'm_b': m_b,
        'm_c': m_c,
        'n_layers': n_layers
    }
    model = utils.build_model_concat_size_and_n_patches_same_number_of_layers(architecture_dict,
                                                                              input_shape,
                                                                              max_number_of_patches)
    return model


def bootstrap_pts_without_mlp_b(n_layers: int, m_a: int, m_c: int,
                                max_number_of_patches: int, input_shape: Tuple[int, int]) -> tf.keras.models.Model:
    model = build_pts_wihout_mlp_b_model(m_a, m_c, n_layers, input_shape, max_number_of_patches)
    return model


def bootstrap_pts_encoder_mlps(hidden_sizes_mlp_a: List[Tuple[int, int]], mlp_a_dropout_rate: float,
                               hidden_sizes_mlp_c: List[Tuple[int, int]], mlp_c_dropout_rate: float,
                               activation: str, input_shape: Tuple[int, int],
                               max_number_of_patches: int) -> tf.keras.models.Model:
    model = build_pts_encoder_mlp_model(hidden_sizes_mlp_a, mlp_a_dropout_rate,
                                         hidden_sizes_mlp_c, mlp_c_dropout_rate,
                                         activation, input_shape, max_number_of_patches)
    return model


model_to_bootstrapper = {
    'patch_to_score_original': bootstrap_patch_to_score_original,  # patch_to_score original model
    'pts_without_mlp_b': bootstrap_pts_without_mlp_b,  # patch_to_score without mlp b,
    'pts_transformer_encoder_mlp': bootstrap_pts_encoder_mlps  # patch_to_score with transformer encoder mlp
}


def build_model_from_configuration(name: str, kwargs: dict) -> tf.keras.models.Model:
    supported_models = list(model_to_bootstrapper.keys())
    if name not in model_to_bootstrapper.keys():
        raise Exception(
            f'model: {name} not supported. supported models: {supported_models}')
    return model_to_bootstrapper[name](**kwargs)
