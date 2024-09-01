import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import paths
from data_preparation.patch_to_score import data_development_utils as development_utils
from data_preparation.ScanNet.db_creation_scanNet_utils import save_as_pickle, load_as_pickle
import data_development_utils as dev_utils
import numpy as np
import protein_level_data_partition_utils as partition_utils
from data_preparation.ScanNet.create_tables_and_weights import cluster_sequences
import tensorflow as tf
   


def create_merged_protein_object_dict():
    merged_dict = {}
    for i in range(len(dev_utils.indexes) - 1):
        d = load_as_pickle(os.path.join(paths.patches_dicts_path, 'proteinObjectsWithEvoluion' + str(i)))
        for key, value in d.items():
            merged_dict[key] = value
    return merged_dict


def create_data_relevant_for_training(max_number_of_components, merged_dict):
    proteins = [protein for _, protein in merged_dict.items()]
    sequences = [protein.get_sequence() for protein in proteins]
    sources = [protein.source for protein in proteins]
    uniprots = [key for key, _ in merged_dict.items()]
    protein_paths = [os.path.join(paths.patches_dicts_path, f'proteinObjectsWithEvoluion{str(i // 1500)}') for i in
                     range(len(uniprots))]

    data_components_flattend, data_protein_size, data_number_of_components, data_components = dev_utils.extract_protein_data(
        proteins,
        max_number_of_components)

    assert (len(uniprots) == len(sequences) == len(protein_paths) == len(
        data_components) == len(data_protein_size) == len(sources) == len(data_number_of_components))
    return uniprots, sequences, protein_paths, data_components_flattend, data_protein_size, data_number_of_components, data_components, sources


def create_patches(all_predictions):
    i = int(sys.argv[1])
    PLDDT_THRESHOLD = 50
    dev_utils.create_patches_dict(i, paths.patches_dicts_path, PLDDT_THRESHOLD, all_predictions)


def create_small_sample_dict(merge_dict):
    small_dict = {}
    for i, (key, value) in enumerate(merge_dict.items()):
        if i == 10:
            break
        small_dict[key] = value
    save_as_pickle(small_dict, os.path.join(paths.patches_dicts_path, 'small_sample_dict.pkl'))


if __name__ == "__main__":
    # CREATE PROTEIN OBJECTS, I'M DOING IT IN BATCHES
    # all_predictions = all_predictions = load_as_pickle(os.path.join(paths.ScanNet_results_path, 'all_predictions_0304_MSA_True.pkl'))
    # create_patches(all_predictions)

    # all_predictions = load_as_pickle(os.path.join(paths.ScanNet_results_path, 'all_predictions_0304_MSA_True.pkl'))
    # print(all_predictions.keys())
    # merged_dict = create_merged_protein_object_dict()
    # save_as_pickle(merged_dict, os.path.join(paths.patches_dicts_path, 'merged_protein_objects_with_evolution.pkl'))
    merged_dict = load_as_pickle(os.path.join(paths.patches_dicts_path, 'merged_protein_objects_with_evolution.pkl'))
    # merged_dict = load_as_pickle(os.path.join(paths.patches_dicts_path, 'proteinObjectsWithEvoluion0'))
    # create_small_sample_dict(merged_dict)
    # merged_dict = load_as_pickle(os.path.join(paths.patches_dicts_path, 'small_sample_dict.pkl'))
    
    # GET RELEVANT INFO FROM PROTEIN OBJECTS

    proteins = [protein for _, protein in merged_dict.items()]
    uniprots, sequences, protein_paths, data_components_flattend, data_protein_size,data_number_of_components, data_components, sources = create_data_relevant_for_training(
       dev_utils.MAX_NUMBER_OF_COMPONENTS, merged_dict)
    # CREATE SCALERS
   
    dev_utils.fit_protein_data(np.array(data_components_flattend), np.array(data_protein_size),  np.array(data_number_of_components),
                              paths.scalers_path, dev_utils.MAX_NUMBER_OF_COMPONENTS)

    # SAVE DATA FOR TRAINING
    save_as_pickle(sources,os.path.join(paths.patch_to_score_data_for_training_path, 'sources.pkl'))
    save_as_pickle(uniprots, os.path.join(paths.patch_to_score_data_for_training_path, 'uniprots.pkl'))
    save_as_pickle(protein_paths,os.path.join(paths.patch_to_score_data_for_training_path, 'protein_paths.pkl'))
    labels = tf.convert_to_tensor([0 if source in dev_utils.NEGATIVE_SOURCES else 1 for source in sources])
    dev_utils.save_as_tensor(labels,os.path.join(paths.patch_to_score_data_for_training_path, 'labels.tf'))
    
    scaled_sizes, scaled_components_list, encoded_components_list = (
    dev_utils.transform_protein_data_list(proteins,
                                              os.path.join(paths.scalers_path, 'scaler_size.pkl'),
                                              os.path.join(paths.scalers_path, 'scaler_components.pkl'),
                                              os.path.join(paths.scalers_path, 'encoder.pkl'),
                                              dev_utils.MAX_NUMBER_OF_COMPONENTS))
    dev_utils.save_as_tensor(scaled_sizes, os.path.join(paths.patch_to_score_data_for_training_path, 'scaled_sizes.tf'))
    dev_utils.save_as_tensor(scaled_components_list, os.path.join(paths.patch_to_score_data_for_training_path, 'scaled_components_list.tf'))
    dev_utils.save_as_tensor(encoded_components_list, os.path.join(paths.patch_to_score_data_for_training_path, 'encoded_components_list.tf'))

    # PARTIOTION THE DATA BY SEQUENCE LIKELIHOOD
    partition_utils.partition_to_folds_and_save(sequences)
    
    # CREATE UNIPROTS SETS
    partition_utils.create_uniprots_sets()
    

    # PLOT DUMMY BASELINE FOR AGGREGATE SCORING FUNCTION
    # plotDummyPRAUC(allPredictions)

    # !!!!
