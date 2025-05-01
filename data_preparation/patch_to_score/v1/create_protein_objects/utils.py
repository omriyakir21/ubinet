import os

import numpy as np

from data_preparation.patch_to_score.v1.schema.protein import Protein
from data_preparation.ScanNet.db_creation_scanNet_utils import save_as_pickle, load_as_pickle


NEGATIVE_SOURCES = set(
    ['Yeast proteome', 'Human proteome', 'Ecoli proteome', 'Celegans proteome', 'Arabidopsis proteome'])

POSITIVE_SOURCES = set(['E1', 'E2', 'E3', 'ubiquitinBinding', 'DUB'])


def create_90_percentile(all_predictions) -> float:
    all_predictions_ubiq = all_predictions['dict_predictions_ubiquitin']
    all_predictions_ubiq_flatten = [value for values_list in all_predictions_ubiq.values() for value in values_list]
    percentile_90 = np.percentile(all_predictions_ubiq_flatten, 90)
    print('percentile_90:', percentile_90)
    return percentile_90


def create_patches_dict(i, dir_path, plddt_threshold, all_predictions,percentile_90,with_pesto):
    indexes_path = os.path.join(dir_path, 'indexes.pkl')
    if not os.path.exists(indexes_path):
        indexes = list(range(0, len(all_predictions['dict_resids']) + 1, 1500)) + [len(all_predictions['dict_resids'])]
        save_as_pickle(indexes, indexes_path)
    indexes = load_as_pickle(indexes_path)
    print(f'len indexes is : {len(indexes)}')
    patches_dict = {}
    all_keys = list(all_predictions['dict_resids'].keys())[indexes[i]:indexes[i + 1]]
    cnt = 0
    for key in all_keys:
        print("i= ", i, " cnt = ", cnt, " key = ", key)
        cnt += 1
        patches_dict[key] = Protein(key, plddt_threshold, all_predictions, percentile_90,with_pesto)
    save_as_pickle(patches_dict, os.path.join(os.path.join(dir_path, 'proteinObjectsWithEvoluion' + str(i))))

 
def extract_protein_data(proteins, max_number_of_components):
    data_components_flattend = []
    data_protein_size = []
    data_number_of_components = []
    data_components = []
    for protein in proteins:
        # Sort components by average_ubiq in descending order and take the top 10
        top_components = sorted(protein.connected_components_tuples, key=lambda x: x[1]['average_scanNet_ubiq'], reverse=True)[
                         :max_number_of_components]
        data_components.append([component[:2] for component in top_components])
        for component in top_components:
            patch_size, patch_dict = component[:2]
            patch_dict_flattened = [float(val) for val in patch_dict.values()]
            patch_flattened = [patch_size] + patch_dict_flattened
            print(f'patch_dict : {patch_dict}')
            print(f'patch_flattened : {patch_flattened}')
            data_components_flattend.append(patch_flattened)
        data_protein_size.append(protein.size)
        data_number_of_components.append(len(top_components))
    return data_components_flattend, data_protein_size, data_number_of_components, data_components

