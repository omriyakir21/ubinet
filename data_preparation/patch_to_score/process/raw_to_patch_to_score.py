import sys
import os
from typing import Tuple

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from data_preparation.patch_to_score import data_development_utils as dev_utils
from data_preparation.ScanNet.db_creation_scanNet_utils import save_as_pickle, load_as_pickle

from data_preparation.patch_to_score.schema.raw_base import RawProtein
from data_preparation.patch_to_score.schema.patch_to_score_base import PatchToScoreProtein


def create_merged_protein_object_dict(dir_path):
    indexes = load_as_pickle(os.path.join(dir_path, 'indexes.pkl'))
    merged_dict = {}
    for i in range(len(indexes) - 1):
        d = load_as_pickle(os.path.join(
            dir_path, 'proteinObjectsWithEvoluion' + str(i)))
        for key, value in d.items():
            merged_dict[key] = value
    return merged_dict


def create_data_relevant_for_training(max_number_of_components, merged_dict, dir_path):
    proteins = [protein for _, protein in merged_dict.items()]
    sequences = [protein.get_sequence() for protein in proteins]
    sources = [protein.source for protein in proteins]
    uniprots = [key for key, _ in merged_dict.items()]
    protein_paths = [os.path.join(dir_path, f'proteinObjectsWithEvoluion{str(i // 1500)}') for i in
                     range(len(uniprots))]

    data_components_flattend, data_protein_size, data_number_of_components, data_components = dev_utils.extract_protein_data(
        proteins,
        max_number_of_components)

    assert (len(uniprots) == len(sequences) == len(protein_paths) == len(
        data_components) == len(data_protein_size) == len(sources) == len(data_number_of_components))
    return uniprots, sequences, protein_paths, data_components_flattend, data_protein_size, data_number_of_components, data_components, sources


def create_patches_dict(i, plddt_threshold, all_predictions, percentile_90, with_pesto) -> dict:
    indexes = list(range(0, len(
        all_predictions['dict_resids']) + 1, 1500)) + [len(all_predictions['dict_resids'])]
    print(f'len indexes is : {len(indexes)}')
    patches_dict = {}
    all_keys = list(all_predictions['dict_resids'].keys())[
        indexes[i]:indexes[i + 1]]
    cnt = 0
    for key in all_keys:
        print("i= ", i, " cnt = ", cnt, " key = ", key)
        cnt += 1
        patches_dict[key] = dev_utils.Protein(
            key, plddt_threshold, all_predictions, percentile_90, with_pesto)
    return patches_dict


def create_patches(all_predictions: dict, percentile_90: float) -> dict:
    i = int(sys.argv[1])  # TODO
    PLDDT_THRESHOLD = 50

    return dev_utils.create_patches_dict(i, PLDDT_THRESHOLD, all_predictions, percentile_90)


def create_small_sample_dict(merge_dict, small_merged_dict_path):
    small_dict = {}
    for i, (key, value) in enumerate(merge_dict.items()):
        if i == 30:
            break
        small_dict[key] = value
    save_as_pickle(small_dict, small_merged_dict_path)
    return small_dict


def add_pesto_predictions(all_predictions_path, pesto_predictions_path):
    pesto_predictions = load_as_pickle(pesto_predictions_path)
    all_predictions = load_as_pickle(all_predictions_path)
    scanNet_keys = all_predictions['dict_resids'].keys()
    pesto_keys = pesto_predictions['protein'].keys()
    intersection = set(scanNet_keys).intersection(set(pesto_keys))
    pesto_predictions = {f'pesto_{class_name}': {sub_k: sub_v for sub_k, sub_v in class_dict.items(
    ) if sub_k in intersection} for class_name, class_dict in pesto_predictions.items()}

    all_predictions.update(pesto_predictions)
    path_with_pesto = all_predictions_path.split(".pkl")[0]+"_with_pesto.pkl"
    save_as_pickle(all_predictions, path_with_pesto)
    return all_predictions


def transform_protein_data_list(proteins, scaler_size, scaler_components, encoder,
                                max_number_of_components, with_pesto):
    scaled_sizes = []
    scaled_components_list = []
    encoded_components_list = []

    for protein in proteins:
        scaled_size_tensor, scaled_components_tensor, encoded_components_tensor = dev_utils.transform_protein_data(
            protein, scaler_size, scaler_components, encoder, max_number_of_components, with_pesto)

        scaled_sizes.append(scaled_size_tensor)
        scaled_components_list.append(scaled_components_tensor)
        encoded_components_list.append(encoded_components_tensor)

    # Print shapes of lists before stacking
    print("scaled_sizes list shape:", len(scaled_sizes),
          scaled_sizes[0].shape if scaled_sizes else None)
    print(f'scaled_sizes is {scaled_sizes}')
    print("scaled_components_list shape:", len(scaled_components_list),
          scaled_components_list[0].shape if scaled_components_list else None)
    print("encoded_components_list shape:", len(encoded_components_list),
          encoded_components_list[0].shape if encoded_components_list else None)
    scaled_sizes = tf.squeeze(tf.convert_to_tensor(scaled_sizes), axis=1)
    encoded_components_list = tf.squeeze(
        tf.convert_to_tensor(encoded_components_list), axis=1)
    return scaled_sizes, tf.convert_to_tensor(scaled_components_list), encoded_components_list


def normalize_data(proteins, scaler_size, scaler_components, encoder):
    scaled_sizes, scaled_components_list, encoded_components_list = transform_protein_data_list(
        proteins,
        scaler_size, scaler_components, encoder,
        dev_utils.MAX_NUMBER_OF_COMPONENTS, with_pesto)

    return scaled_sizes, scaled_components_list, encoded_components_list


def save_data(data_for_training_folder_path, proteins, sequences, sources, uniprots, protein_paths, scaled_sizes, scaled_components_list, encoded_components_list):
    dev_utils.save_as_tensor(scaled_sizes, os.path.join(
        data_for_training_folder_path, 'scaled_sizes.tf'))
    dev_utils.save_as_tensor(scaled_components_list, os.path.join(
        data_for_training_folder_path, 'scaled_components_list.tf'))
    dev_utils.save_as_tensor(encoded_components_list, os.path.join(
        data_for_training_folder_path, 'encoded_components_list.tf'))
    save_as_pickle(sources, os.path.join(
        data_for_training_folder_path, 'sources.pkl'))
    save_as_pickle(uniprots, os.path.join(
        data_for_training_folder_path, 'uniprots.pkl'))
    save_as_pickle(protein_paths, os.path.join(
        data_for_training_folder_path, 'protein_paths.pkl'))
    labels = tf.convert_to_tensor(
        [0 if source in dev_utils.NEGATIVE_SOURCES else 1 for source in sources])
    dev_utils.save_as_tensor(labels, os.path.join(
        data_for_training_folder_path, 'labels.tf'))


def load_predictions(predictions_path: str) -> dict:
    predictions = load_as_pickle(predictions_path)
    return predictions


def create_90_percentile(all_predictions: dict, based_on: str) -> float:
    all_predictions_ubiq = all_predictions[based_on]
    all_predictions_ubiq_flatten = [
        value for values_list in all_predictions_ubiq.values() for value in values_list]
    percentile_90 = np.percentile(all_predictions_ubiq_flatten, 90)
    return percentile_90


def fit_protein_data(all_data_components, all_data_protein_size, all_data_number_of_components,
                     max_number_of_components) -> Tuple[StandardScaler, StandardScaler, OneHotEncoder]:
    # Fit the scalers
    scaler_size = StandardScaler()
    scaler_components = StandardScaler()
    scaler_size.fit(all_data_protein_size.reshape(-1, 1))
    scaler_components.fit(all_data_components)

    # Fit the encoder
    encoder = OneHotEncoder(sparse_output=False, categories=[
                            range(max_number_of_components + 1)])
    encoder.fit(all_data_number_of_components.reshape(-1, 1))

    return scaler_size, scaler_components, encoder


def create_pts_proteins(predictions_path: str, save_dir_path: str,
                        pdb_path: str, scannet_scores_path: str, pesto_scores_path: str) -> None:
    """
    predictions_path = '/home/iscb/wolfson/doririmon/home/order/ubinet/repo/ubinet/results/ScanNet/all_predictions_0304_MSA_True_with_pesto.pkl'
    """

    patches_dict_folder_path = os.path.join(save_dir_path, 'patches_dicts')
    data_for_training_folder_path = os.path.join(
        save_dir_path, 'data_for_training')
    for dir_path in [patches_dict_folder_path, data_for_training_folder_path]:
        os.makedirs(dir_path, exist_ok=True)

    all_predictions = load_predictions(predictions_path)
    percentile_90 = create_90_percentile(
        all_predictions, based_on='dict_predictions_ubiquitin')
    # TODO: handle chunking & slurm cpus
    patches = create_patches(all_predictions, percentile_90)
    # TODO: save patches as 'merged_protein_objects.pkl'
    merged_dict = patches  # TODO: this is true after merging..
    proteins = [protein for _, protein in merged_dict.items()]
    uniprots, sequences, protein_paths, data_components_flattend, data_protein_size, data_number_of_components, data_components, sources = create_data_relevant_for_training(
        dev_utils.MAX_NUMBER_OF_COMPONENTS, merged_dict, patches_dict_folder_path)
    scaler_size, scaler_components, encoder = fit_protein_data(np.array(data_components_flattend),
                                                               np.array(
                                                                   data_protein_size),
                                                               np.array(
                                                                   data_number_of_components),
                                                               dev_utils.MAX_NUMBER_OF_COMPONENTS)
    
    scaled_sizes, scaled_components_list, encoded_components_list = normalize_data(
        proteins, scaler_size, scaler_components, encoder)

    save_data(data_for_training_folder_path, proteins, sequences, sources, uniprots,
              protein_paths, scaled_sizes, scaled_components_list, encoded_components_list)
