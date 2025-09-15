import os
from typing import List, Tuple
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np
import tensorflow as tf
from utils import save_as_pickle, load_as_pickle, create_paths, save_as_tensor
from data_preparation.patch_to_score.v1.schema.base import PatchToScoreProteinChain


def extract_protein_data(protein_chains: List[PatchToScoreProteinChain]):
    data_components_flattend = []
    data_protein_size = []
    data_number_of_components = []
    for protein in protein_chains:
        for patch in protein.patches:
            patch_size = patch.size
            # TODO: we don't want to average anymore
            patch_dict = patch.get_average_scores_dict()
            patch_dict_flattened = [float(val) for val in patch_dict.values()]
            patch_flattened = [patch_size] + patch_dict_flattened
            data_components_flattend.append(patch_flattened)
        data_protein_size.append(protein.number_of_amino_acids)
        data_number_of_components.append(protein.number_of_patches)
    return data_components_flattend, data_protein_size, data_number_of_components


def fit_protein_data(all_data_components, all_data_protein_size, all_data_number_of_components, dir_path,
                     max_number_of_components):
    # Fit the scalers
    scaler_size = StandardScaler()
    scaler_components = StandardScaler()
    scaler_size.fit(all_data_protein_size.reshape(-1, 1))
    scaler_components.fit(all_data_components)

    # Fit the encoder
    encoder = OneHotEncoder(sparse_output=False, categories=[
                            range(max_number_of_components + 1)])
    encoder.fit(all_data_number_of_components.reshape(-1, 1))

    # Save the scalers and encoder
    save_as_pickle(scaler_size, os.path.join(dir_path, 'scaler_size.pkl'))
    save_as_pickle(scaler_components, os.path.join(
        dir_path, 'scaler_components.pkl'))
    save_as_pickle(encoder, os.path.join(dir_path, 'encoder.pkl'))


def transform_protein_data(protein: PatchToScoreProteinChain,
                           scaler_size: StandardScaler,
                           scaler_components: StandardScaler,
                           encoder: OneHotEncoder,
                           max_number_of_components: int) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    scaled_size = scaler_size.transform(
        np.array([protein.number_of_amino_acids]).reshape(-1, 1))

    # Extract, scale, and pad the components
    patches = protein.patches
    component_size = len(
        protein.amino_acids[0].get_scores_dict()) + 1  # +1 for the size
    if len(patches) == 0:
        protein_components_scaled = np.zeros((0, component_size))
    else:
        protein_components_flattend = []
        for patch in patches:

            patch_size = patch.size
            # TODO: we don't want to average anymore
            patch_dict = patch.get_average_scores_dict()
            patch_dict_flattened = [float(val) for val in patch_dict.values()]
            patch_flattened = [patch_size] + patch_dict_flattened
            protein_components_flattend.append(patch_flattened)
        protein_components_flattend = np.array(protein_components_flattend)
        protein_components_scaled = scaler_components.transform(
            protein_components_flattend)
    if len(protein_components_scaled) < max_number_of_components:
        padding = ((0, max_number_of_components -
                   len(protein_components_scaled)), (0, 0))
        protein_components_scaled = np.pad(
            protein_components_scaled, padding, mode='constant', constant_values=0)

    # Encode the number of components
    encoded_components = encoder.transform(
        np.array([len(patches)]).reshape(-1, 1))

    # Convert to tensors
    scaled_size_tensor = tf.convert_to_tensor(scaled_size)
    scaled_components_tensor = tf.convert_to_tensor(protein_components_scaled)
    encoded_components_tensor = tf.convert_to_tensor(encoded_components)

    return scaled_size_tensor, scaled_components_tensor, encoded_components_tensor


def transform_protein_data_list(proteins: List[PatchToScoreProteinChain],
                                scaler_size_path: str,
                                scaler_components_path: str,
                                encoder_path: str,
                                max_number_of_components: int) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    scaler_size = load_as_pickle(scaler_size_path)
    scaler_components = load_as_pickle(scaler_components_path)
    encoder = load_as_pickle(encoder_path)
    scaled_sizes = []
    scaled_components_list = []
    encoded_components_list = []

    for protein in tqdm(proteins):
        scaled_size_tensor, scaled_components_tensor, encoded_components_tensor = transform_protein_data(
            protein, scaler_size, scaler_components, encoder, max_number_of_components)

        scaled_sizes.append(scaled_size_tensor)
        scaled_components_list.append(scaled_components_tensor)
        encoded_components_list.append(encoded_components_tensor)

    scaled_sizes = tf.squeeze(tf.convert_to_tensor(scaled_sizes), axis=1)
    encoded_components_list = tf.squeeze(
        tf.convert_to_tensor(encoded_components_list), axis=1)
    return scaled_sizes, tf.convert_to_tensor(scaled_components_list), encoded_components_list


def normalize_and_save_data(data_for_training_folder_path: str,
                            scalers_folder_path: str,
                            proteins: List[PatchToScoreProteinChain],
                            max_number_of_components: int):
    scaled_sizes, scaled_components_list, encoded_components_list = (
        transform_protein_data_list(proteins,
                                    os.path.join(
                                        scalers_folder_path, 'scaler_size.pkl'),
                                    os.path.join(
                                        scalers_folder_path, 'scaler_components.pkl'),
                                    os.path.join(
                                        scalers_folder_path, 'encoder.pkl'),
                                    max_number_of_components))

    save_as_tensor(scaled_sizes, os.path.join(
        data_for_training_folder_path, 'scaled_sizes.tf'))
    save_as_tensor(scaled_components_list, os.path.join(
        data_for_training_folder_path, 'scaled_components_list.tf'))
    save_as_tensor(encoded_components_list, os.path.join(
        data_for_training_folder_path, 'encoded_components_list.tf'))

    return scaled_sizes, scaled_components_list, encoded_components_list


def main(scalers_folder_path: str,
         data_for_training_folder_path: str,
         protein_chains: List[PatchToScoreProteinChain], max_number_of_components: int):
    print('extracting protein data')
    data_components_flattend, data_protein_size, data_number_of_components = extract_protein_data(
        protein_chains)
    print('fitting')
    create_paths(scalers_folder_path)
    fit_protein_data(np.array(data_components_flattend), np.array(data_protein_size),  np.array(data_number_of_components),
                     scalers_folder_path, max_number_of_components)
    print('normalizing')
    scaled_sizes, scaled_components_list, encoded_components_list = normalize_and_save_data(data_for_training_folder_path,
                                                                                            scalers_folder_path,
                                                                                            protein_chains,
                                                                                            max_number_of_components)
    return scaled_sizes, scaled_components_list, encoded_components_list
