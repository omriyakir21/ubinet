import os
import numpy as np
from utils import create_paths, save_as_pickle, load_as_pickle


def create_90_percentile(all_predictions: dict, by: str = 'dict_predictions_ubiquitin') -> float:
    all_predictions_ubiq = all_predictions[by]
    all_predictions_ubiq_flatten = [
        value for values_list in all_predictions_ubiq.values() for value in values_list]
    percentile_90 = np.percentile(all_predictions_ubiq_flatten, 90)
    print('computed 90 percentile:', percentile_90)
    return percentile_90


def main(all_predictions_path: str,
         save_dir_path: str) -> None:
    """
    Create protein objects.

   :param str all_predictions_path: path to a pickle file containing all predictions in the form of a dictionary
   :param str save_dir_path: path to a directory where the protein objects will be saved
   :param str sources_path: path to a directory where the source .pdb files are saved
   :param list uniprot_names: list of unitprot names to create protein objects for
   :param bool with_pesto: should use pesto predictions
   :return: None
   :rtype: None
    """
    percentile_90_dir = os.path.join(save_dir_path, 'global_values')
    create_paths(percentile_90_dir)
    all_predictions = load_as_pickle(all_predictions_path)
    percentile_90 = create_90_percentile(all_predictions)
    save_as_pickle(percentile_90, os.path.join(
        percentile_90_dir, 'percentile_90.pkl'))
