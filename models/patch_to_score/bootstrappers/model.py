import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
import utils
import tensorflow as tf

def bootstrap_patch_to_score_original(n_layers: int, m_a: int, m_b: int, m_c: int, 
                                      max_number_of_patches: int, input_shape: int) -> tf.keras.Model:
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

model_to_bootstrapper = {
    'patch_to_score_original': bootstrap_patch_to_score_original
}

def build_model_from_configuration(name: str, kwargs: dict) -> tf.keras.Model:
    supported_models = list(model_to_bootstrapper.keys())
    if name not in model_to_bootstrapper.keys():
        raise Exception(f'model: {name} not supported. supported models: {supported_models}')
    return model_to_bootstrapper[name](**kwargs)