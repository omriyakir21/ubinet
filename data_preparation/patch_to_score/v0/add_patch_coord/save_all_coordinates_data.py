# -- imports --
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

import sys
sys.path.append('data_preparation/patch_to_score/v0/')

import pickle
from typing import Tuple, Dict, List, Union
import os
from tqdm import tqdm
import numpy as np
from utils import create_paths
from data_preparation.patch_to_score.v0.data_development_utils import Protein


# -- constants --
data_for_training_dir_path = 'datasets/patch_to_score/data_for_training'
source_dataset_dir_path = f'{data_for_training_dir_path}/03_04_with_pesto'
target_dataset_dir_path = f'{data_for_training_dir_path}/03_04_with_pesto_and_coord'

MAX_NUMBER_OF_COMPONENTS = 10


# -- utils --
# copied from data_development_utils.py
# modified a bit
# TODO: the main risk here: sorted might change order from original creation 
def extract_protein_data(proteins, max_number_of_components):
    residue_indexes = []
    for protein in proteins:
        # Sort components by average_ubiq in descending order and take the top 10
        top_components = sorted(protein.connected_components_tuples, key=lambda x: x[1]['average_scanNet_ubiq'], reverse=True)[
                         :max_number_of_components]
        residue_indexes.append([component[2] for component in top_components])
    return residue_indexes


# -- load --
def load_data() -> Tuple[Dict[str, Protein], Dict[str, List[List[int]]]]:
    print('loading data')
    with open('datasets/patch_to_score/patches_dicts/03_04_with_pesto/merged_protein_objects.pkl', 'rb') as f:
        merged_protein_objects = pickle.load(f)
        
    data_set_res_indexes = {k: extract_protein_data([v], MAX_NUMBER_OF_COMPONENTS)[0] for k, v in merged_protein_objects.items()}
    print(f'loaded {len(data_set_res_indexes)} proteins')
    return merged_protein_objects, data_set_res_indexes

 # -- coord --
def get_all_patches_ca_coordinates(merged_protein_objects: Dict[str, Protein], data_set_res_indexes: Dict[str, List[List[int]]]) -> Dict[str, List[Union[np.array, None]]]:
    print('fetching all patches CA coordinates')
    data_set_coordinates = dict()

    for uniprot_id, protein_object in tqdm({k: v for k, v in merged_protein_objects.items() if k in data_set_res_indexes}.items()):
        data_set_coordinates[uniprot_id] = []
        structure = protein_object.get_structure()
        model = structure.child_list[0]
        assert len(model) == 1
        protein_indexes = data_set_res_indexes[uniprot_id]
        
        for i, patch_indexes in enumerate(protein_indexes):
            data_set_coordinates[uniprot_id].append([])
            for residue_index in patch_indexes:
                for chain in model:
                    coord = chain.child_list[residue_index]['CA'].coord  # TODO: is this logic correct?
                    data_set_coordinates[uniprot_id][i].append(coord)
        
            if len(data_set_coordinates[uniprot_id][i]) == 0:
                data_set_coordinates[uniprot_id][i] = None
            else:
                data_set_coordinates[uniprot_id][i] = np.vstack(data_set_coordinates[uniprot_id][i])
    
    return data_set_coordinates
            

# -- save --
def save_all_coordinates_data(data_set_coordinates: Dict[str, List[Union[np.array, None]]], 
                              data_set_res_indexes: Dict[str, List[List[int]]],
                              target_dataset_dir_path: str):
    """
    Save the coordinates data to a pickle file.
    """
    print('saving all coordinates data')
    create_paths(target_dataset_dir_path)

    save_items = {
        f'{target_dataset_dir_path}/all_patches_ca_coordinates.pkl': data_set_coordinates,
        f'{target_dataset_dir_path}/data_set_res_indexes.pkl': data_set_res_indexes
    }
    for save_path, obj_to_save in save_items.items():
        if not os.path.exists(save_path):
            with open(save_path, 'wb') as f:
                pickle.dump(obj_to_save, f)
                
        print(f"saved to {save_path}")


# -- main --
if __name__ == '__main__':
    merged_protein_objects, data_set_res_indexes = load_data()
    data_set_coordinates = get_all_patches_ca_coordinates(merged_protein_objects, data_set_res_indexes)
    save_all_coordinates_data(data_set_coordinates, data_set_res_indexes, target_dataset_dir_path)
