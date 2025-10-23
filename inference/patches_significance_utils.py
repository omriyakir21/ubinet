
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import paths
from tqdm import tqdm
from data_preparation.ScanNet.db_creation_scanNet_utils import save_as_pickle, load_as_pickle, \
    THREE_LETTERS_TO_SINGLE_AA_DICT, aa_out_of_chain
import json
import numpy as np
from models.patch_to_score.bootstrappers.model import build_model_from_configuration
import tensorflow as tf
from typing import Dict, List, Tuple, Optional
from data_preparation.patch_to_score.v0.data_development_utils import Protein, MAX_NUMBER_OF_COMPONENTS
import pandas as pd
from results.patch_to_score.cut_protein import extract_patch_from_pdb
from concurrent.futures import ThreadPoolExecutor


def batch_k_computation(predictions, training_ub_ratio, with_class_weights=False):
    if with_class_weights:
        training_ub_ratio = 0.5
    val = 1 - predictions
    # To avoid division by zero, we return np.nan for entries where val equals 0
    K = np.where(val == 0, np.nan, ((1 - training_ub_ratio) * predictions) / (training_ub_ratio * val))
    return K

def batch_prediction_function_using_bayes_factor_computation(priorUb, KValues):
    """
    Vectorized computation of final predictions using Bayes factor.
    
    Parameters:
        priorUb: float or numpy array of prior probabilities.
        KValues: numpy array of K values with shape (batch_size,) or (batch_size, 1).
    
    Returns:
        numpy array of final predictions.
    """
    finalPredictions = (KValues * priorUb) / ((KValues * priorUb) + (1 - priorUb))
    return finalPredictions

def str_patch_from_list_indexes(protein, list_locations):
    structure = protein.get_structure()
    model = structure.child_list[0]
    assert (len(model) == 1)
    for chain in model:
        residues = aa_out_of_chain(chain)
        myList = [THREE_LETTERS_TO_SINGLE_AA_DICT[residues[index].resname] + str(residues[index].id[1]) for index in
              list_locations]
    strPatch = ','.join(myList)
    return strPatch

def create_training_ub_ratio(folds_training_dicts_path:str, ub_ratios_dir:str):
    folds_training_dicts = load_as_pickle(folds_training_dicts_path)
    labels = []
    for fold_index in range(len(folds_training_dicts)):
        fold = folds_training_dicts[fold_index]
        labels = fold['labels_train']
        positives_ratio = np.sum(labels == 1) / len(labels)
        ub_ratio_fold_path = os.path.join(ub_ratios_dir, f'ub_ratio_fold_{fold_index}.pkl')
        save_as_pickle(positives_ratio, ub_ratio_fold_path)


def calculate_significances_util(model, batch, ub_ratio_path):
    ub_ratio = load_as_pickle(ub_ratio_path)
    components_batch, sizes_batch, num_patches_batch, original_preds,_ = batch
    batch_size = components_batch.shape[0]
    feat_dim = components_batch.shape[-1]
    actual_npatches_full = tf.argmax(num_patches_batch, axis=-1, output_type=tf.int32)
    actual_npatches_full = tf.reshape(actual_npatches_full,[-1,1])  # shape (N,)
    indices = np.arange(10)  # shape (10,)
    effective_npatches_full = tf.broadcast_to(actual_npatches_full, (batch_size, 10))
    effective_npatches_full = np.where(indices < actual_npatches_full, actual_npatches_full - 1, actual_npatches_full)
    one_hot_effective_npatches = tf.one_hot(effective_npatches_full, depth=11)
    # Initialize per-batch accumulator lists

    mod_preds_batch = []
    for j in range(10):
        first_part = components_batch[:, :j, :]
        second_part = components_batch[:, j+1:, :]
        removed = tf.concat([first_part, second_part], axis=1)
        # Append a zero row to recover shape (10, feat_dim)
        zero_patch = tf.zeros((components_batch.shape[0], 1, feat_dim), dtype=components_batch.dtype)
        modified = tf.concat([removed, zero_patch], axis=1)
        # Predict using the model on the modified batch
        mod_preds = model.predict([modified, sizes_batch, one_hot_effective_npatches[:, j, :]])
        mod_preds_batch.append(mod_preds)
    mod_preds_batch = np.concatenate(mod_preds_batch, axis=1)
    mod_K_values_batch = batch_k_computation(mod_preds_batch, ub_ratio)
    mod_inference_preds_batch = batch_prediction_function_using_bayes_factor_computation(0.05, mod_K_values_batch)
    original_K_values_batch = batch_k_computation(original_preds, ub_ratio)
    original_inference_preds_batch = batch_prediction_function_using_bayes_factor_computation(0.05, original_K_values_batch)
    significances_batch = np.broadcast_to(original_inference_preds_batch, (original_inference_preds_batch.shape[0], 10)) - mod_inference_preds_batch

    
    return original_preds, original_inference_preds_batch,original_K_values_batch \
            , mod_preds_batch, mod_inference_preds_batch, mod_K_values_batch \
            , significances_batch

def get_top_patches_indices(protein):
    components = protein.connected_components_tuples
    if len(components) == 0:
        return None
    sorted_components = sorted(components, key=lambda x: x[1]['average_scanNet_ubiq'], reverse=True)
    top_component_sets = [component[2] for component in
                          sorted_components[:min(MAX_NUMBER_OF_COMPONENTS, len(components))]]
    return top_component_sets

def create_sub_fold_training_dicts_test(folds_training_dicts:list,uniprot:str):
    for fold in folds_training_dicts:
        if uniprot in fold['uniprots_test']:
            components = fold['components_test']
            sizes = fold['sizes_test']
            num_patches = fold['num_patches_test']
            labels = fold['labels_test']
            uniprots = fold['uniprots_test']
            index = list(uniprots).index(uniprot)
            return {
                'components_test': components[index:index+1],
                'sizes_test': sizes[index:index+1],
                'num_patches_test': num_patches[index:index+1],
                'labels_test': labels[index:index+1],
                'uniprots_test': uniprots[index:index+1]
            }
    raise ValueError(f'Uniprot {uniprot} not found in any fold test sets')



def create_str_patches_and_Significances(folds_training_dicts_path: str, model_dir: str, protein_locations_path: str,ub_ratios_dir: str) -> Dict[str, np.ndarray]:
    folds_training_dicts = load_as_pickle(folds_training_dicts_path)
    proteins = load_as_pickle(protein_locations_path)
    with open(f'{model_dir}/configuration.json', 'rb') as f:
        configuration = json.load(f)
    model_configuration = configuration['model']
    model = build_model_from_configuration(**model_configuration)
    data_dict = {
        'all_original_inference_predictions': [],
        'all_original_predictions': [],
        'all_original_K_values': [],
        'all_mod_preds': [],
        'all_mod_inference_predictions': [],
        'all_mod_K_values': [],
        'all_significants': [],
        'all_uniprots': [],
        'all_str_patches': [],
        'all_labels': []
    }


    # Loop over folds
    for fold_index in tqdm(range(len(folds_training_dicts)), desc="Processing Folds"):
        ub_ratio_path = os.path.join(ub_ratios_dir, f'ub_ratio_fold_{fold_index}.pkl')
        model.load_weights(f'{model_dir}/fold_{fold_index}/model.keras')
        fold = folds_training_dicts[fold_index]
        #Debug for F4JIP6
        # if 'F4JIP6' not in fold['uniprots_test']:
        #     print(f'Uniprot F4JIP6 not found in fold {fold_index} test set, skipping this fold.')
        #     continue
        # fold = create_sub_fold_training_dicts_test(folds_training_dicts, 'F4JIP6')
        components_fold = fold['components_test']    # Tensor of shape (N, 10, feat_dim)
        num_patches_fold = fold['num_patches_test']     # Tensor of shape (N, ...)
        sizes_fold = fold['sizes_test']                 # Tensor of shape (N, ...)
        uniprots_fold = fold['uniprots_test']
        labels_fold = fold['labels_test']
        # original_preds_fold = np.load(os.path.join(fold_model_dir, 'test', 'predictions.npy'))
        data_dict['all_labels'].extend(labels_fold)
        original_preds_fold = model.predict([components_fold, sizes_fold,num_patches_fold])
        # Create a dataset for batching the fold data.
        dataset = tf.data.Dataset.from_tensor_slices(
            (components_fold, sizes_fold, num_patches_fold, original_preds_fold, uniprots_fold)
        ).batch(512)
        # Process the fold by batching
        for batch in tqdm(dataset,desc=f"Processing Fold {fold_index} Batches"):
            original_preds, original_inference_preds_batch,original_K_values_batch \
            , mod_preds_batch, mod_inference_preds_batch, mod_K_values_batch \
            , significances_batch = calculate_significances_util(model, batch,ub_ratio_path)
        
            sorted_indices_desc_batch = np.argsort(significances_batch, axis=1)[:, ::-1]
            sorted_mod_preds_batch = np.take_along_axis(mod_preds_batch, sorted_indices_desc_batch, axis=1)
            sorted_mod_inference_preds_batch = np.take_along_axis(mod_inference_preds_batch, sorted_indices_desc_batch, axis=1)
            sorted_mod_K_values_batch = np.take_along_axis(mod_K_values_batch, sorted_indices_desc_batch, axis=1)
            sorted_significances_batch = np.take_along_axis(significances_batch, sorted_indices_desc_batch, axis=1)

            uniprots_batch = batch[4].numpy()
            proteins_batch = [proteins[uniprot.decode('utf-8')] for uniprot in uniprots_batch]
            str_patches_batch = []
            

            for i in range(len(proteins_batch)):
                protein = proteins_batch[i]
                top_component_sets = get_top_patches_indices(protein)
                if top_component_sets is None:
                    str_patches_batch.append([])
                    continue
                sorted_component_sets = []
                cnt = 0
                while cnt < MAX_NUMBER_OF_COMPONENTS:
                    if sorted_indices_desc_batch[i][cnt] < len(top_component_sets):
                        sorted_component_sets.append(top_component_sets[sorted_indices_desc_batch[i][cnt]])
                    cnt+=1
                str_patches_protein = []
                for j in range(len(sorted_component_sets)):
                    str_patches_protein.append(str_patch_from_list_indexes(protein, sorted_component_sets[j]))
                str_patches_batch.append(str_patches_protein)
            data_dict['all_original_predictions'].extend(original_preds)
            data_dict['all_original_inference_predictions'].extend(original_inference_preds_batch)
            data_dict['all_original_K_values'].extend(original_K_values_batch)
            data_dict['all_mod_preds'].extend(sorted_mod_preds_batch)
            data_dict['all_mod_inference_predictions'].extend(sorted_mod_inference_preds_batch)
            data_dict['all_mod_K_values'].extend(sorted_mod_K_values_batch)
            data_dict['all_significants'].extend(sorted_significances_batch)
            data_dict['all_uniprots'].extend(uniprots_batch)
            data_dict['all_str_patches'].extend(str_patches_batch)
    for key,val in data_dict.items():
        if not key == 'all_str_patches':
            data_dict[key] = np.stack(val,axis=0)
    return data_dict

def filter_all_predictions_by_source(source:str,
                input_csv_path:str,
                top_k:int,
                output_path:str):

    df = pd.read_csv(input_csv_path)
    filtered_df = df[df['source'] == source]
    top_k_df = filtered_df.nlargest(top_k, 'inference_prediction')

    # Save the filtered DataFrame to a new CSV file
    top_k_df.to_csv(output_path, index=False)
    print(f'Saved top {top_k} of {source}CSV to {output_path}')
    return top_k_df

def create_csv_add_sources_and_significance_ll(data_dict_path:str,proteins_path: str, output_csv:str):
    proteins = load_as_pickle(proteins_path)
    data_dict = load_as_pickle(data_dict_path)
    all_uniprots = data_dict['all_uniprots']
    sources = [proteins[uniprot.decode('utf-8')].source for uniprot in all_uniprots]
    # Create a DataFrame from the data dictionary

    rows = []
    for i in range(len(data_dict['all_str_patches'])):
        row = {'uniprot': all_uniprots[i].decode('utf-8'),
               'predictions': float(data_dict['all_original_predictions'][i]),
                'inference_prediction': float(data_dict['all_original_inference_predictions'][i]),
                'source': sources[i]}

        str_patches = data_dict['all_str_patches'][i]
        significances = data_dict['all_significants'][i]
        cnt = 0
        for j in range(MAX_NUMBER_OF_COMPONENTS):
            if significances[j] == 0:
                continue
                
            row[f'significance{cnt}'] = float(significances[j])
            row[f'significance_ll_{cnt}'] = calculate_log_likelihood_significance(
                inference_prediction=float(data_dict['all_original_inference_predictions'][i]),
                significance=float(significances[j]))
            row[f'str_patch{cnt}'] = str_patches[cnt]

            cnt += 1
        for j in range(cnt, MAX_NUMBER_OF_COMPONENTS):
            row[f'significance{j}'] = ''
            row[f'str_patch{j}'] = ''
            row[f'significance_ll_{j}'] = ''
        rows.append(row)
    df = pd.DataFrame(rows)
    # sort the rows by inference_prediction in descending order
    df = df.sort_values(by='inference_prediction', ascending=False)
    # Set float format to 4 decimal places
    pd.options.display.float_format = '{:.4f}'.format
    # Save the DataFrame to a CSV file
    df.to_csv(output_csv, index=False)
    print(f'Saved CSV to {output_csv}')


def calculate_log_likelihood_significance(inference_prediction:float, significance:float) -> float:
    log_likelihood_inference_with_patch = np.log(inference_prediction / (1 - inference_prediction))
    log_likelihood_inference_without_patch = np.log((inference_prediction - significance) / 
                                                    (1 - inference_prediction + significance))
    return log_likelihood_inference_with_patch - log_likelihood_inference_without_patch


def create_substructures_for_filtered(csv_file:str,substructures_dir:str,all_predictions_ubiq:list,patch_number:int,source:str,output_path:str):
    os.makedirs(substructures_dir, exist_ok=True)
    substructures_source_dir = os.path.join(substructures_dir, source.split(" ")[0])
    os.makedirs(substructures_source_dir, exist_ok=True)
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Iterate through each row in the DataFrame
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        uniprot_id = row['uniprot']
        str_patch = row[f'str_patch{patch_number}']
        all_predictions_ubiq_uniprot = all_predictions_ubiq[uniprot_id]
        # Call extract_patch_from_pdb
        # Assuming input_file is handled later, using a placeholder for now
        input_file = os.path.join(paths.AFDB_source_patch_to_score_path,source.split(" ")[0],f'{uniprot_id}.pdb')
        if str_patch == 'nan' or pd.isna(str_patch) or str_patch=='' :
            df.drop(index, inplace=True)
            continue
        residue_count = extract_patch_from_pdb(id = uniprot_id, patch = str_patch, input_file = input_file
        , output_dir = substructures_source_dir, all_predictions_ubiq = all_predictions_ubiq_uniprot)
        # Remove the row if residue_count is greater than 300
        if residue_count > 300:
            df.drop(index, inplace=True)
    # Save the filtered DataFrame to the output CSV file
    df.to_csv(output_path, index=False)
    print(f'Saved filtered data to {output_path}')