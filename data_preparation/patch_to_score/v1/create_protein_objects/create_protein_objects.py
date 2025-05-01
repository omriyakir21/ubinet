import sys
import os
from typing import List
from data_preparation.patch_to_score.v1.create_protein_objects.utils import create_patches_dict, create_90_percentile
from data_preparation.ScanNet.db_creation_scanNet_utils import save_as_pickle, load_as_pickle


def create_merged_protein_object_dict(dir_path):
    indexes = load_as_pickle(os.path.join(dir_path, 'indexes.pkl'))
    merged_dict = {}
    for i in range(len(indexes) - 1):
        d = load_as_pickle(os.path.join(
            dir_path, 'proteinObjectsWithEvoluion' + str(i)))
        for key, value in d.items():
            merged_dict[key] = value
    return merged_dict


def create_paths(*paths: List[str]) -> None:
    for path in paths:
        os.makedirs(path, exist_ok=True)


def main(all_predictions_path: str, save_dir_path: str, with_pesto: bool):
    """
    Create protein objects.

   :param str all_predictions_path: path to a pickle file containing all predictions in the form of a dictionary
   :param str save_dir_path: path to a directory where the protein objects will be saved
   :param bool with_pesto: whether to all_predictions object includes pesto predictions or not
   :return: a list of protein objects
   :rtype: List[Protein]
    """
    create_paths(save_dir_path)
    percentile_90_path = os.path.join(save_dir_path, 'percentile_90.pkl')
    merged_dict_path = os.path.join(
        save_dir_path, 'merged_protein_objects.pkl')

    all_predictions = load_as_pickle(all_predictions_path)

    if not os.path.exists(percentile_90_path):
        percentile_90 = create_90_percentile(all_predictions)
        save_as_pickle(percentile_90, os.path.join(
            save_dir_path, 'percentile_90.pkl'))
    else:
        percentile_90 = load_as_pickle(percentile_90_path)

    # CREATING THE PATCHES IN BATCHES OF 1500 PROTEINS. SEE SCRIPTS/RUN_DATA_DEVELOPMENT.SH
    # WE CAN RUN THIS ON CPU'S (CAN DO MULTIPLE AT A TIME)
    i = int(sys.argv[1])  # TODO
    plddt_threshold = 50
    create_patches_dict(i, save_dir_path, plddt_threshold, all_predictions, percentile_90, with_pesto)

    if not os.path.exists(merged_dict_path):
        merged_dict = create_merged_protein_object_dict(save_dir_path)
        save_as_pickle(merged_dict, merged_dict_path)
    else:
        merged_dict = load_as_pickle(merged_dict_path)
