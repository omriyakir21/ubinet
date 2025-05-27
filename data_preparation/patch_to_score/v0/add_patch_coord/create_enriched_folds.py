import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

from typing import Dict, List, Union, Tuple
import numpy as np
from utils import load_as_pickle, save_as_pickle
from copy import deepcopy


data_for_training_dir_path = 'datasets/patch_to_score/data_for_training'
source_dataset_dir_path = f'{data_for_training_dir_path}/03_04_with_pesto'
target_dataset_dir_path = f'{data_for_training_dir_path}/03_04_with_pesto_and_coord'


def create_coordinates_matrix(uniprots: List[str], patches_ca_avg_coordinates: Dict[str, Union[np.ndarray, None]]) -> np.ndarray:
    data = [patches_ca_avg_coordinates[uniprot_id] for uniprot_id in uniprots]
    padded_data = []
    for item in data:
        if item is None:
            padded = np.zeros((10, 3))
        else:
            padded = np.vstack([item, np.zeros((10 - item.shape[0], 3))])
        padded_data.append(padded)
    return np.stack(padded_data)


def load_data() -> Tuple[List[dict], Dict[str, List[Union[np.ndarray, None]]]]:
    print('loading data')
    folds_training_dicts = load_as_pickle(f'{source_dataset_dir_path}/folds_training_dicts.pkl')  # TODO: maybe use curated dataset
    all_patches_ca_coordinates = load_as_pickle(f'{target_dataset_dir_path}/all_patches_ca_coordinates.pkl')
    return folds_training_dicts, all_patches_ca_coordinates


def enrich_folds_with_coordinates(folds_training_dicts: List[dict], all_patches_ca_coordinates: Dict[str, List[Union[np.ndarray, None]]]) -> List[dict]:    
    print('enriching folds with coordinates')
    res = deepcopy(folds_training_dicts)
    for fold in res:
        for set_name in ['train', 'validation', 'test']:
            uniprots = fold[f'uniprots_{set_name}']
            patches_ca_avg_coordinates = {uniprot_id: [np.average(c, axis=0) if c is not None else None for c in coordinates] for uniprot_id, coordinates in all_patches_ca_coordinates.items()}
            patches_ca_avg_coordinates = {k: np.vstack(v) if (len(v) > 0) else None for k, v in patches_ca_avg_coordinates.items()}
            coordinates_matrix = create_coordinates_matrix(uniprots, patches_ca_avg_coordinates)
            fold[f'coordinates_{set_name}'] = tf.convert_to_tensor(coordinates_matrix)
    return res


if __name__ == '__main__':
    folds_training_dicts, all_patches_ca_coordinates = load_data()
    enriched_folds = enrich_folds_with_coordinates(folds_training_dicts, all_patches_ca_coordinates)
    print('saving enriched folds')
    save_as_pickle(folds_training_dicts, f'{target_dataset_dir_path}/folds_training_dicts.pkl')
    print('saved enriched folds')
