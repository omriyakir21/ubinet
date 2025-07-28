import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
import paths
from data_preparation.patch_to_score.v0 import data_development_utils as development_utils
from data_preparation.ScanNet.db_creation_scanNet_utils import save_as_pickle, load_as_pickle
import data_development_utils as dev_utils
import numpy as np
import protein_level_data_partition_utils as partition_utils
from data_preparation.ScanNet.create_tables_and_weights import cluster_sequences
import tensorflow as tf
   


def create_merged_protein_object_dict(dir_path):
    indexes = load_as_pickle(os.path.join(dir_path, 'indexes.pkl'))
    merged_dict = {}
    for i in range(len(indexes) - 1):
        d = load_as_pickle(os.path.join(dir_path, 'proteinObjectsWithEvoluion' + str(i)))
        for key, value in d.items():
            merged_dict[key] = value
    return merged_dict


def create_data_relevant_for_training(max_number_of_components, merged_dict,dir_path):
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


def create_patches(all_predictions,dir_path,percentile_90, patch_index: int):
    PLDDT_THRESHOLD = 50

    dev_utils.create_patches_dict(patch_index, dir_path, PLDDT_THRESHOLD, all_predictions,percentile_90)


def create_small_sample_dict(merge_dict,small_merged_dict_path):
    small_dict = {}
    for i, (key, value) in enumerate(merge_dict.items()):
        if i == 30:
            break
        small_dict[key] = value
    save_as_pickle(small_dict, small_merged_dict_path)
    return small_dict



def add_pesto_predictions(path_with_pesto,pesto_predictions_path):
    pesto_predictions = load_as_pickle(pesto_predictions_path)
    all_predictions = load_as_pickle(all_predictions_path)  
    scanNet_keys = all_predictions['dict_resids'].keys()
    pesto_keys = pesto_predictions['protein'].keys()
    intersection = set(scanNet_keys).intersection(set(pesto_keys))
    pesto_predictions = {f'pesto_{class_name}': {sub_k: sub_v for sub_k, sub_v in class_dict.items() if sub_k in intersection} for class_name, class_dict in pesto_predictions.items()}
    
    all_predictions.update(pesto_predictions)
    save_as_pickle(all_predictions, path_with_pesto)
    return all_predictions

def normalize_and_save_data(data_for_training_folder_path, proteins, sequences, sources, uniprots, protein_paths):
    scaled_sizes, scaled_components_list, encoded_components_list = (
        dev_utils.transform_protein_data_list(proteins,
                                                os.path.join(scalers_folder_path, 'scaler_size.pkl'),
                                                os.path.join(scalers_folder_path, 'scaler_components.pkl'),
                                                os.path.join(scalers_folder_path, 'encoder.pkl'),
                                                dev_utils.MAX_NUMBER_OF_COMPONENTS,with_pesto))
    
    dev_utils.save_as_tensor(scaled_sizes, os.path.join(data_for_training_folder_path, 'scaled_sizes.tf'))
    dev_utils.save_as_tensor(scaled_components_list, os.path.join(data_for_training_folder_path, 'scaled_components_list.tf'))
    dev_utils.save_as_tensor(encoded_components_list, os.path.join(data_for_training_folder_path, 'encoded_components_list.tf'))
    save_as_pickle(sources,os.path.join(data_for_training_folder_path, 'sources.pkl'))
    save_as_pickle(uniprots, os.path.join(data_for_training_folder_path, 'uniprots.pkl'))
    save_as_pickle(protein_paths,os.path.join(data_for_training_folder_path, 'protein_paths.pkl'))
    labels = tf.convert_to_tensor([0 if source in dev_utils.NEGATIVE_SOURCES else 1 for source in sources])
    dev_utils.save_as_tensor(labels,os.path.join(data_for_training_folder_path, 'labels.tf'))

if __name__ == "__main__":
    plan_dict = {
        'date': '03_04',
        'with_pesto': True,
        'recreate_with_pesto': False,
        'all_predictions_path': os.path.join(paths.ScanNet_results_path, 'all_predictions_0304_MSA_True.pkl'),
        'pesto_predictions_path': '/home/iscb/wolfson/doririmon/home/order/ubinet/pesto/C_structured/PeSToIntegration/assets/data/pesto_inference_outputs/dict_predictions_pesto.pkl',
        
        'create_percentile': True,
        'create_patches': False,
        'merge_patches': False,
        'fetch_and_partition': False,
        'create_dummy_predictor': False  # stay False for now
    }
    DATE = plan_dict['date']
    with_pesto = plan_dict['with_pesto']
    with_pesto_addition = '_with_pesto' if with_pesto else ''
    training_name = f'{DATE}{with_pesto_addition}_trial'
    all_predictions_path = plan_dict['all_predictions_path']
    path_with_pesto = all_predictions_path.split(".pkl")[0]+"_with_pesto.pkl"
    
    
    patches_dict_folder_path = os.path.join(paths.patches_dicts_path, f'{training_name}')
    os.makedirs(patches_dict_folder_path, exist_ok=True)
    data_for_training_folder_path = os.path.join(paths.patch_to_score_data_for_training_path, f'{training_name}')
    os.makedirs(data_for_training_folder_path, exist_ok=True)
    pesto_predictions_path = plan_dict['pesto_predictions_path']
    merged_dict = None
    if plan_dict['recreate_with_pesto']:
        all_predictions = add_pesto_predictions(path_with_pesto,pesto_predictions_path)
    
    elif plan_dict['with_pesto']:
        all_predictions = load_as_pickle(path_with_pesto)
        
    else:
        all_predictions = load_as_pickle(all_predictions_path)
    
    if plan_dict['create_percentile']:
        # SAVE THE 90'TH PERCENTILE OF UBIQUITIN BINDING PREDICTIONS FROM SCANNET,
        # WE WILL USE IT LATER AS A THRESHOLD FOR AN AMINO ACID IN ORDER TO BE IN A PATCH 
        percentile_90_path = os.path.join(patches_dict_folder_path, 'percentile_90.pkl')
        if not os.path.exists(percentile_90_path):
            dev_utils.create_90_percentile(all_predictions, percentile_90_path)
    percentile_90 = load_as_pickle(percentile_90_path)

    # CREATING THE PATCHES IN BATCHES OF 1500 PROTEINS. SEE SCRIPTS/RUN_DATA_DEVELOPMENT.SH 
    # WE CAN RUN THIS ON CPU'S (CAN DO MULTIPLE AT A TIME)
    if plan_dict['create_patches']:
        create_patches(all_predictions,patches_dict_folder_path,percentile_90,with_pesto, patch_index=int(sys.argv[1]))   

    #MERGE THE PATCHES AFTER CREATING THEM
    if plan_dict['merge_patches']:
        merged_dict = create_merged_protein_object_dict(patches_dict_folder_path)
        merged_dict_path = os.path.join(patches_dict_folder_path, 'merged_protein_objects.pkl')
        if not os.path.exists(merged_dict_path):
            save_as_pickle(merged_dict, merged_dict_path)

    #CREATE SMALL SAMPLE FOR TESTING
    # small_sample_dict_path = os.path.join(patches_dict_folder_path, 'small_sample_dict.pkl')
    # if not os.path.exists(small_sample_dict_path):
    #     merged_dict = create_small_sample_dict(merged_dict,small_sample_dict_path)
    # else:
    #     merged_dict = load_as_pickle(small_sample_dict_path)
    
    
    if plan_dict['fetch_and_partition']:
        merged_dict = load_as_pickle(merged_dict_path)
        
        # GET RELEVANT INFO FROM PROTEIN OBJECTS
        proteins = [protein for _, protein in merged_dict.items()]
        uniprots, sequences, protein_paths, data_components_flattend, data_protein_size,data_number_of_components, data_components, sources = create_data_relevant_for_training(
        dev_utils.MAX_NUMBER_OF_COMPONENTS, merged_dict,patches_dict_folder_path)
        
        # CREATE AND FIT SCALERS
        patch_to_score_model_path =os.path.join(paths.patch_to_score_model_path, f'{training_name}') 
        scalers_folder_path = os.path.join(patch_to_score_model_path, 'scalers')
        os.makedirs(scalers_folder_path, exist_ok=True)
        dev_utils.fit_protein_data(np.array(data_components_flattend), np.array(data_protein_size),  np.array(data_number_of_components),
                                scalers_folder_path, dev_utils.MAX_NUMBER_OF_COMPONENTS)


        # SAVE DATA FOR TRAINING
        normalize_and_save_data(data_for_training_folder_path, proteins, sequences, sources, uniprots, protein_paths)

        # PARTIOTION THE DATA BY SEQUENCE LIKELIHOOD
        partition_utils.partition_to_folds_and_save(sequences,data_for_training_folder_path)
        
        # CREATE UNIPROTS SETS (WE ARE USING IT LATER IN RESULTS ANALYSIS)
        partition_utils.create_uniprots_sets(data_for_training_folder_path)
        

    # PLOT DUMMY BASELINE (protein prediction is the prediction of the highest amino acid prediction) FOR AGGREGATE SCORING FUNCTION
    if plan_dict['create_dummy_predictor']:
        dev_utils.plot_dummy_prauc(all_predictions)
