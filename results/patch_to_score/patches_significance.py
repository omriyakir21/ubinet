import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import numpy as np
import paths
from data_preparation.ScanNet.db_creation_scanNet_utils import save_as_pickle, load_as_pickle, \
    THREE_LETTERS_TO_SINGLE_AA_DICT, aa_out_of_chain
from data_preparation.patch_to_score.data_development_utils import Protein, MAX_NUMBER_OF_COMPONENTS, \
    transform_protein_data_list,load_as_tensor,NEGATIVE_SOURCES
from models.patch_to_score.patch_to_score_MLP_utils import k_computation, \
    prediction_function_using_bayes_factor_computation
from results.patch_to_score.patch_to_score_result_analysis import get_best_architecture_models_path,create_best_architecture_results_dir
import tensorflow as tf
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import auc
from typing import Dict, List
import pdb
from cut_protein import extract_patch_from_pdb
from models.structural_aligners.TM_align.TM_aligner import align_uniprots_with_target
import results.patch_to_score.chimera_script as chimera_script


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


def find_model_number(uniprot, uniprots_sets):
    for i in range(5):
        if uniprot in uniprots_sets[i]:
            return i
    return None


def predict_for_protein(uniprot, scaled_size, scaled_components, encoded_components, models):
    uniprots_sets = load_as_pickle(os.path.join(paths.patch_to_score_data_for_training_path, 'uniprots_sets.pkl'))
    modelIndex = find_model_number(uniprot, uniprots_sets)
    # print(f'modelIndex: {modelIndex}')
    if modelIndex is None:        
        yhat = np.mean(np.concatenate([model.predict([scaled_components, scaled_size, encoded_components]) for model in models]),
            axis=0)

    else:
        model = models[modelIndex]
        # pdb.set_trace()
        # print(f'scaled_components shape: {scaled_components.shape}, scaled_size shape: {scaled_size.shape}, encoded_components shape: {encoded_components.shape}')
        # print(f'scaled_components: {scaled_components}, scaled_size: {scaled_size}, encoded_components: {encoded_components}')
        yhat = model.predict([scaled_components, scaled_size, encoded_components])[0]
        
        # print(f'yhat: {yhat}')
    return yhat


def get_info_util(all_predictions, plddt_threshold,models_dir,results_dir):

    uniprots = list(all_predictions['dict_resids'].keys())
    proteins = [Protein(uniprot, plddt_threshold, all_predictions) for uniprot in uniprots]
    scaled_sizes, scaled_components_list, encoded_components_list = (
        transform_protein_data_list(proteins,
                                    os.path.join(paths.scalers_path, 'scaler_size.pkl'),
                                    os.path.join(paths.scalers_path, 'scaler_components.pkl'),
                                    os.path.join(paths.scalers_path, 'encoder.pkl'),
                                    MAX_NUMBER_OF_COMPONENTS))
    models = []
    models_dir_path = get_best_architecture_models_path(models_dir,results_dir)
    for filename in os.listdir(models_dir_path):
        if filename.endswith('.keras'):
            models.append(tf.keras.models.load_model(os.path.join(models_dir_path, filename)))
    return uniprots, proteins, scaled_sizes, scaled_components_list, encoded_components_list, models


def get_patch_to_score_prediction_for_uniprots(all_predictions, plddt_threshold):
    uniprots, proteins, scaled_sizes, scaled_components_list, encoded_components_list, models = get_info_util(
        all_predictions, plddt_threshold,paths.with_MSA_50_plddt_0304_models_dir,paths.with_MSA_50_plddt_0304_results_dir)

    predictions = []
    for i in range(len(proteins)):
        yhat = predict_for_protein(uniprots[i], scaled_sizes[i, :], tf.reshape(scaled_components_list[i, :, :],(1,MAX_NUMBER_OF_COMPONENTS,4)),
                                   encoded_components_list[i, :], models)
        print(yhat)
        predictions.append(yhat)
    return predictions


def create_training_ub_ratio(labels,all_predictions_dir):
    ub_ratio = float(tf.reduce_mean(tf.cast(labels,tf.float32)).numpy())
    print(f'ub_ratio: {ub_ratio}')
    save_as_pickle(ub_ratio, os.path.join(all_predictions_dir, 'ub_ratio.pkl'))


def get_top_patches_indices(protein):
    components = protein.connected_components_tuples
    if len(components) == 0:
        return None
    sorted_components = sorted(components, key=lambda x: x[1], reverse=True)
    top_component_sets = [component[4] for component in
                          sorted_components[:min(MAX_NUMBER_OF_COMPONENTS, len(components))]]
    return top_component_sets


def create_str_patches_and_Significance_for_protein(prediction, uniprot, scaled_size, scaled_components,
                                                    encoded_components, models, ub_Ratio, protein):
    k_prediction = k_computation(prediction, ub_Ratio)
    inferece_prediction = prediction_function_using_bayes_factor_computation(0.05, k_prediction)
    print(f'inferece_prediction: {inferece_prediction}')
    significances = []
    str_patches = []
    number_of_patches = tf.argmax(tf.reshape(encoded_components,(-1)))
    if number_of_patches == 0:
        str_patches = [None for i in range(10)]
        significances = ['' for i in range(10)]
        return str_patches, significances

    indices = tf.argmax(encoded_components, axis=-1)
    indices_subtracted = tf.maximum(indices - 1, 0)
    one_hot_tensor = tf.one_hot(indices_subtracted, depth=encoded_components.shape[-1])
    for i in range(number_of_patches):
        without_patch = tf.concat([scaled_components[:i], scaled_components[i + 1:]], axis=0)
        padding_needed = MAX_NUMBER_OF_COMPONENTS - without_patch.shape[0] 
        paddings =[[0,padding_needed], [0,0]]
        padded_tensor = tf.reshape(tf.pad(without_patch, paddings, 'CONSTANT',constant_values=0),(1,10,4))
        yhat = predict_for_protein(uniprot, scaled_size, padded_tensor, one_hot_tensor, models)
        k_value = k_computation(yhat, ub_Ratio)
        inference_yhat = prediction_function_using_bayes_factor_computation(0.05, k_value)
        print(f'inference_yhat: {inference_yhat}')
        significances.append(inferece_prediction - inference_yhat)

    sorted_significance_indices = np.argsort(np.array(significances))
    descending_indices = sorted_significance_indices[::-1]
    sorted_significance = [significances[i] for i in descending_indices]
    sorted_significance_padded = sorted_significance + [0] * (MAX_NUMBER_OF_COMPONENTS - len(sorted_significance))
    top_component_sets = get_top_patches_indices(protein)
    sorted_component_sets = [top_component_sets[i] for i in descending_indices]
    for i in range(len(sorted_component_sets)):
        str_patches.append(str_patch_from_list_indexes(protein, sorted_component_sets[i]))
    print(f'sorted_significance_padded: {sorted_significance_padded}')
    return sorted_significance_padded, str_patches

import pdb
def create_str_patches_and_Significances(all_predictions, plddt_threshold, ub_ratio):
    # pdb.set_trace()
    significances_lists = []
    str_patches_lists = []
    predictions = get_patch_to_score_prediction_for_uniprots(all_predictions, plddt_threshold)
    uniprots, proteins, scaled_sizes, scaled_components_list, encoded_components_list, models = get_info_util(
        all_predictions, plddt_threshold,paths.with_MSA_50_plddt_0304_models_dir,paths.with_MSA_50_plddt_0304_results_dir)
    for i in range(len(proteins)):
        print(f'uniprot: {proteins[i].uniprot_name}',flush=True)
        print(f'prediction: {predictions[i]}',flush=True)
        print(f'scaled_size: {scaled_sizes[i, :]}',flush=True)
        print(f'scaled_components: {scaled_components_list[i, :, :]}',flush=True)
        print(f'encoded_components: {encoded_components_list[i, :]}',flush=True)

        sorted_significance_padded, str_patches = create_str_patches_and_Significance_for_protein(float(predictions[i]), uniprots[i], scaled_sizes[i, :],
                                                        scaled_components_list[i, :, :], encoded_components_list[i, :],
                                                        models, ub_ratio, proteins[i])
        significances_lists.append(sorted_significance_padded)
        str_patches_lists.append(str_patches)
    return uniprots,significances_lists, str_patches_lists

def create_str_patches_and_Significance10(significance, sorted_locations_cutted, protein):
    structure = protein.getStructure()
    model = structure.child_list[0]
    assert (len(model) == 1)
    for chain in model:
        residues = aa_out_of_chain(chain)
    strPatches = [None for _ in range(10)]
    significance10 = [None for _ in range(10)]
    for i in range(len(significance)):
        strPatch = str_patch_from_list_indexes(residues, sorted_locations_cutted[significance[i][0]])
        strPatches[i] = strPatch
        significance10[i] = significance[i][1]
    return strPatches, significance10


def filter_dicts_by_keys(data: Dict[str, Dict[str, any]], keys: List[str]) -> Dict[str, Dict[str, any]]:
    """
    Filter a dictionary of dictionaries to include only the specified keys in each inner dictionary.

    Args:
    data (Dict[str, Dict[str, any]]): Dictionary of dictionaries to filter.
    keys (List[str]): List of keys to retain in each inner dictionary.

    Returns:
    Dict[str, Dict[str, any]]: New dictionary of dictionaries containing only the specified keys.
    """
    return {
        outer_key: {inner_key: value for inner_key, value in inner_dict.items() if inner_key in keys}
        for outer_key, inner_dict in data.items()
    }

def create_uniprots_inference_prediction_csv(ub_Ratio):
    best_architecture_models_path = get_best_architecture_models_path(paths.with_MSA_50_plddt_0304_models_dir,paths.with_MSA_50_plddt_0304_results_dir)
    best_architecture_results_dir = create_best_architecture_results_dir(best_architecture_models_path,paths.with_MSA_50_plddt_0304_results_dir)    
    folds_training_dicts = load_as_pickle(os.path.join(paths.patch_to_score_data_for_training_path,
                                                                        'folds_traning_dicts.pkl'))
    all_uniprots = []
    inference_prediction_dict = {}
    for i in range(len(folds_training_dicts)):
        dict_for_training = folds_training_dicts[i]
        uniprots_test = dict_for_training['uniprots_test']
        all_uniprots.extend(uniprots_test)
    predictions_test = np.load(os.path.join(best_architecture_models_path,'predictions_test.npy'))
    assert(len(all_uniprots) == predictions_test.size)
    for i in range (predictions_test.size):
        k_prediction = k_computation( predictions_test[i], ub_Ratio)
        inferece_prediction = prediction_function_using_bayes_factor_computation(0.05, k_prediction)
        inference_prediction_dict[all_uniprots[i]] = inferece_prediction
    #crearte a .csv of the dictionary
    inference_prediction_df = pd.DataFrame(list(inference_prediction_dict.items()),columns = ['uniprot','inference_prediction'])
    inference_prediction_df.to_csv(os.path.join(best_architecture_results_dir,'inference_prediction.csv'),index=False)

    
def create_csv_with_uniprots_data_and_inference():
    # Read the CSV files into DataFrames
    uniprot_data = pd.read_csv(os.path.join(paths.patch_to_score_results_path,'uniprot_data.csv'))
    best_architecture_models_path = get_best_architecture_models_path(paths.with_MSA_50_plddt_0304_models_dir,paths.with_MSA_50_plddt_0304_results_dir)
    best_architecture_results_dir = create_best_architecture_results_dir(best_architecture_models_path,paths.with_MSA_50_plddt_0304_results_dir)  
    inference_prediction = pd.read_csv(os.path.join(best_architecture_results_dir,'inference_prediction.csv'))

    # Merge the DataFrames on the 'UniProt ID' and 'uniprot' columns
    merged_data = pd.merge( inference_prediction, uniprot_data, left_on='uniprot', right_on='UniProt ID')

    merged_data = merged_data.drop(columns=['UniProt ID'])

    # Save the merged DataFrame to a new CSV file
    merged_data.to_csv(os.path.join(best_architecture_results_dir,'uniprots_data_with_predictions.csv'), index=False)

def add_sources_to_csv(all_predictions_path):
    all_predictions = load_as_pickle(all_predictions_path)
    sources_dict = all_predictions['dict_sources']
    
    # Convert sources_dict to DataFrame
    sources_df = pd.DataFrame(list(sources_dict.items()), columns=['uniprot', 'source'])

    best_architecture_models_path = get_best_architecture_models_path(paths.with_MSA_50_plddt_0304_models_dir, paths.with_MSA_50_plddt_0304_results_dir)
    best_architecture_results_dir = create_best_architecture_results_dir(best_architecture_models_path, paths.with_MSA_50_plddt_0304_results_dir)
    uniprots_data_with_predictions = pd.read_csv(os.path.join(best_architecture_results_dir, 'uniprots_data_with_predictions.csv'))
    
    # Merge the DataFrames
    merged_data = pd.merge(uniprots_data_with_predictions, sources_df, on='uniprot', how='right')
    
    # Save the merged DataFrame to a new CSV file
    merged_data.to_csv(os.path.join(best_architecture_results_dir, 'data_predictions_sources.csv'), index=False)

def process_top_uniprots_by_source(file_path, source, output_file,all_predictions_path):
    # Load the CSV file
    df = pd.read_csv(file_path)
    
    # Filter by source
    filtered_df = df[df['source'] == source]
    
    # Sort by inference_prediction and select top 500
    top_500_df = filtered_df.nlargest(500, 'inference_prediction')
    print(f'top_500_df: {top_500_df}')

    uniprots = top_500_df['uniprot'].tolist()
    print(f'uniprots: {uniprots}')

    ub_ratio = load_as_pickle(os.path.join(paths.patch_to_score_data_for_training_path, 'ub_ratio.pkl'))
    all_predictions = load_as_pickle(os.path.join(paths.ScanNet_results_path, 'all_predictions_0304_MSA_True.pkl')) 
    all_predictions_sub = filter_dicts_by_keys(all_predictions, uniprots)
    
    # Calculate significances and strPatches
    uniprots_ordered,significances_lists, str_patches_lists = create_str_patches_and_Significances(all_predictions_sub, 50, ub_ratio)   
    
    rows = []

        # Iterate over the uniprots and their corresponding significances and str_patches
    for uniprot, significances, str_patches in zip(uniprots_ordered, significances_lists, str_patches_lists):
        # Create a dictionary for each uniprot
        row = {'uniprot': uniprot}
        for i, (significance, str_patch) in enumerate(zip(significances, str_patches), start=1):
            row[f'significance{i}'] = significance
            row[f'str_patch{i}'] = str_patch
        # Append the dictionary to the list of rows
        rows.append(row)

    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(rows)

    # Merge the DataFrames
    top_500_df = pd.merge(top_500_df, df, on='uniprot')

    # Save to a new CSV file
    top_500_df.to_csv(output_file, index=False)

def calculate_log_likelihood_significance(inference_prediction, significance):
    log_likelihood_inference_with_patch = np.log(inference_prediction / (1 - inference_prediction))
    log_likelihood_inference_without_patch = np.log((inference_prediction - significance) / 
                                                    (1 - inference_prediction + significance))
    return log_likelihood_inference_with_patch - log_likelihood_inference_without_patch

def calculate_significance_ll(input_csv, output_csv):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(input_csv)
    
    # Iterate through each significance column
    for i in range(1, 11):
        significance_col = f'significance{i}'
        
        # Calculate significance_ll_{index} using the utility function
        df[f'significance_ll_{i}'] = df.apply(lambda row: calculate_log_likelihood_significance(row['inference_prediction'], row[significance_col]), axis=1)
        
        # Insert the new column right after the respective significance column
        col_index = df.columns.get_loc(significance_col) + 1
        cols = df.columns.tolist()
        cols.insert(col_index, cols.pop(cols.index(f'significance_ll_{i}')))
        df = df[cols]
    
    # Save the updated DataFrame to a new CSV file
    df.to_csv(output_csv, index=False)

def filter_top_examples(input_csv, output_csv):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(input_csv)
    
    # Filter rows where str_patch1 has fewer than 6 amino acids
    df = df[df['str_patch1'].apply(lambda x: len(x.split(',')) >= 6)]
    
    # Filter rows where significance_ll_1 is lower than ln(2)
    df = df[df['significance_ll_1'] >= np.log(2)]
    
    # Save the filtered DataFrame to the output CSV file
    df.to_csv(output_csv, index=False)

def create_substructures_for_filtered(csv_file, output_dir,substructures_dir,all_predictions_ubiq,patch_number):
    if not os.path.exists(substructures_dir):
        os.makedirs(substructures_dir)
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Iterate through each row in the DataFrame
    for index, row in df.iterrows():
        # Extract uniprot and str_patch1
        uniprot_id = row['uniprot']
        str_patch = row['str_patch1']
        source = row['source']
        organism = source.split()[0]
        all_predictions_ubiq_uniprot = all_predictions_ubiq[uniprot_id]
        # Call extract_patch_from_pdb
        # Assuming input_file is handled later, using a placeholder for now
        input_file = os.path.join(paths.AFDB_source_patch_to_score_path,organism,f'{uniprot_id}.pdb')
        residue_count = extract_patch_from_pdb(uniprot_id, str_patch, patch_number, input_file, substructures_dir,all_predictions_ubiq_uniprot)
        # Remove the row if residue_count is greater than 300
        if residue_count > 300:
            df.drop(index, inplace=True)
    # Save the filtered DataFrame to the output CSV file
    output_path = os.path.join(output_dir, f'data_predictions_significances_final_{organism}.csv')
    df.to_csv(output_path, index=False)

def tm_align_substructures(substructures_dir,aligned_pdbs_dir):
    pdb_chain_table_path = os.path.join(paths.structural_aligners_path,'pdb_chain_table_uniprot_with_classes.csv')
    if not os.path.exists(aligned_pdbs_dir):
        os.makedirs(aligned_pdbs_dir)
    # Iterate through each substructure file in the substructures directory
    for file in os.listdir(substructures_dir):
        # Call TM-align
        # Assuming input_file is handled later, using a placeholder for now
        target_pdb = os.path.join(substructures_dir, file)
        align_uniprots_with_target(target_pdb, pdb_chain_table_path,aligned_pdbs_dir,substructures_dir)

def add_structure_info(input_folder, input_csv, output_csv,patch_number):
    # Read the input CSV file
    df = pd.read_csv(input_csv)
    prefix = f'patch{patch_number}_'
    # Initialize new columns
    new_columns_per_patch = [
        'pdb_id_e1', 'tm_score_e1', 'rmsd_e1', 'seq_identity_e1',
        'pdb_id_e2', 'tm_score_e2', 'rmsd_e2', 'seq_identity_e2',
        'pdb_id_e3|e4', 'tm_score_e3|e4', 'rmsd_e3|e4', 'seq_identity_e3|e4',
        'pdb_id_deubiquitylase', 'tm_score_deubiquitylase', 'rmsd_deubiquitylase', 'seq_identity_deubiquitylase',
        'pdb_id_other', 'tm_score_other', 'rmsd_other', 'seq_identity_other'
    ]
    new_columns = [f'{prefix}{col}' for col in new_columns_per_patch]
    
    for col in new_columns:
        df[col] = None
    
    # Loop through each uniprot ID
    for index, row in df.iterrows():
        uniprot = row['uniprot']
        sub_folder = os.path.join(input_folder, f"{uniprot}_patch{patch_number}")
        structure_csv = os.path.join(sub_folder, f"{uniprot}_patch{patch_number}_top_scores.csv")
        
        if os.path.exists(structure_csv):
            structure_df = pd.read_csv(structure_csv)
            
            # Extract information for each class
            for class_name in ['e1', 'e2', 'e3|e4', 'deubiquitylase','other']:
                class_row = structure_df[structure_df['class'] == class_name]
                if not class_row.empty:
                    df.at[index, f'patch{patch_number}_pdb_id_{class_name}'] = class_row['pdb_id'].values[0]
                    df.at[index, f'patch{patch_number}_tm_score_{class_name}'] = class_row['tm_score'].values[0]
                    df.at[index, f'patch{patch_number}_rmsd_{class_name}'] = class_row['rmsd'].values[0]
                    df.at[index, f'patch{patch_number}_seq_identity_{class_name}'] = class_row['seq_identity'].values[0]
        
    df.to_csv(output_csv, index=False)


def change_bfactor_single_chain_structure(structure, chain_id, new_bfactors):
    for chain in structure[0]:
        if chain.id == chain_id:
            for residue, new_bfactor in zip(chain, new_bfactors):
                residue['CA'].bfactor = new_bfactor
    return structure

if __name__ == '__main__':
    # all_predictions = load_as_pickle(os.path.join(paths.ScanNet_results_path, 'all_predictions_0304_MSA_True.pkl'))
    # labels = load_as_tensor(os.path.join(paths.patch_to_score_data_for_training_path, 'labels.tf'),out_type=tf.int32)
    # create_training_ub_ratio(labels,paths.patch_to_score_data_for_training_path)
    # ub_ratio = load_as_pickle(os.path.join(paths.patch_to_score_data_for_training_path, 'ub_ratio.pkl'))
        # all_predictions_sub = filter_dicts_by_keys(all_predictions, uniprots)
    # # save_as_pickle(all_predictions_sub, os.path.join(paths.with_MSA_50_plddt_0304_results_dir, 'all_predictions_sub.pkl'))
    # all_predictions_sub = load_as_pickle(os.path.join(paths.with_MSA_50_plddt_0304_results_dir, 'all_predictions_sub.pkl'))
    # significances_lists, str_patches_lists = create_str_patches_and_Significances(all_predictions_sub, 50, ub_ratio)
    # print(f'significances_lists: {significances_lists}')
    # print(f'str_patches_lists: {str_patches_lists}')
    # create_uniprots_inference_prediction_csv(ub_ratio)
    # create_csv_with_uniprots_data_and_inference()
    # add_sources_to_csv(os.path.join(paths.ScanNet_results_path, 'all_predictions_0304_MSA_True.pkl'))

         
    # create csv file for source
    best_architecture_models_path = get_best_architecture_models_path(paths.with_MSA_50_plddt_0304_models_dir, paths.with_MSA_50_plddt_0304_results_dir)
    best_architecture_results_dir = create_best_architecture_results_dir(best_architecture_models_path,paths.with_MSA_50_plddt_0304_results_dir)    
    # for source in list(NEGATIVE_SOURCES):    
    #     file_path = os.path.join(best_architecture_results_dir, 'data_predictions_sources.csv')
    #     output_file_path = os.path.join(best_architecture_results_dir, f'data_predictions_significances_{source}.csv')
    #     all_predictions_path = os.path.join(paths.ScanNet_results_path, 'all_predictions_0304_MSA_True.pkl')
    #     process_top_uniprots_by_source(file_path, source, output_file_path,all_predictions_path)

    # add log likelihood difference significance columns
    input_path = os.path.join(best_architecture_results_dir, f'data_predictions_significances_Human proteome.csv')
    # calculate_significance_ll(input_path, input_path)    

    #filter top examples
    output_path = os.path.join(best_architecture_results_dir, f'data_predictions_significances_filtered_Human proteome.csv')
    # filter_top_examples(input_path, output_path)
    patch_number = 1
    substructures_dir = os.path.join(best_architecture_results_dir, 'substructures_chainsaw',f'patch_{patch_number}')
    
    # all_predictions_ubiq = all_predictions['dict_predictions_ubiquitin']

    # create_substructures_for_filtered(output_path, best_architecture_results_dir,substructures_dir,all_predictions_ubiq,patch_number)
    aligned_pdbs_dir = os.path.join(best_architecture_results_dir, 'aligned_chainsaw_pdbs_with_ubiqs',f'patch_{patch_number}')
    # tm_align_substructures(substructures_dir,aligned_pdbs_dir)

    # #add structure info to final table

    input_csv = os.path.join(best_architecture_results_dir, 'data_predictions_significances_final_Human.csv')
    output_csv = os.path.join(best_architecture_results_dir, 'data_predictions_significances_final_Human_with_tm_info.csv')
    add_structure_info(aligned_pdbs_dir, input_csv, output_csv,patch_number)

    patch_number = 2
    substructures_dir = os.path.join(best_architecture_results_dir, 'substructures_chainsaw',f'patch_{patch_number}')
    aligned_pdbs_dir = os.path.join(best_architecture_results_dir, 'aligned_chainsaw_pdbs_with_ubiqs',f'patch_{patch_number}')
    input_csv = os.path.join(best_architecture_results_dir, 'data_predictions_significances_final_Human_with_tm_info.csv')
    output_csv = os.path.join(best_architecture_results_dir, 'data_predictions_significances_final_Human_with_tm_info.csv')
    add_structure_info(aligned_pdbs_dir, input_csv, output_csv,patch_number)



