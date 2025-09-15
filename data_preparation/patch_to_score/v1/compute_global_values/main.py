import os
import numpy as np
from utils import save_as_pickle, load_as_pickle


def create_90_percentile(all_predictions: dict, by: str = 'dict_predictions_ubiquitin') -> float:
    all_predictions_ubiq = all_predictions[by]
    all_predictions_ubiq_flatten = [
        value for values_list in all_predictions_ubiq.values() for value in values_list]
    percentile_90 = np.percentile(all_predictions_ubiq_flatten, 90)
    print('computed 90 percentile:', percentile_90)
    return percentile_90


def main(all_predictions: dict,
         save_dir_path: str,
         should_override: bool) -> float:
    """
    Create protein objects.

   :param str all_predictions: all model predictions over all proteins
   :param str save_dir_path: path to a directory where the protein objects will be saved
   :param bool should_override: should override existing global values
   :return: percentile_90: the 90th percentile of the scannet ubiquitin binding score
   :rtype: float
    """
    save_path = os.path.join(save_dir_path, 'percentile_90.pkl')
    if os.path.exists(save_path) and (not should_override):
        percentile_90 = load_as_pickle(save_path)
        print('loaded pre-calculated 90 percentile:', percentile_90)
        return percentile_90
    percentile_90 = create_90_percentile(all_predictions)
    save_as_pickle(percentile_90, save_path)
    return percentile_90
