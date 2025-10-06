import json
import os
import sys
from tqdm import tqdm
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import inference.patches_significance_utils as psu
import numpy as np
import paths
from data_preparation.ScanNet.db_creation_scanNet_utils import save_as_pickle, load_as_pickle
import tensorflow as tf
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import auc
from typing import Dict, List
import pdb

from models.structural_aligners.TM_align.TM_aligner import align_uniprots_with_target
import results.patch_to_score.chimera_script as chimera_script
from inference.fetch_uniprot_data import uniprot_to_csv


def find_model_number(uniprot, uniprots_sets):
    for i in range(5):
        if uniprot in uniprots_sets[i]:
            return i
    return None


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


# def calculate_significance_ll(input_csv, output_csv):
#     # Read the CSV file into a DataFrame
#     df = pd.read_csv(input_csv)
    
#     # Iterate through each significance column
#     for i in range(1, 11):
#         significance_col = f'significance{i}'
        
#         # Calculate significance_ll_{index} using the utility function
#         df[f'significance_ll_{i}'] = df.apply(lambda row: calculate_log_likelihood_significance(row['inference_prediction'], row[significance_col]), axis=1)
        
#         # Insert the new column right after the respective significance column
#         col_index = df.columns.get_loc(significance_col) + 1
#         cols = df.columns.tolist()
#         cols.insert(col_index, cols.pop(cols.index(f'significance_ll_{i}')))
#         df = df[cols]
    
#     # Save the updated DataFrame to a new CSV file
#     df.to_csv(output_csv, index=False)

def filter_top_examples(input_csv, output_csv):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(input_csv)
    
    # Filter rows where str_patch1 has fewer than 6 amino acids
    df = df[df['str_patch1'].apply(lambda x: len(x.split(',')) >= 6)]
    
    # Filter rows where significance_ll_1 is lower than ln(2)
    df = df[df['significance_ll_1'] >= np.log(2)]
    
    # Save the filtered DataFrame to the output CSV file
    df.to_csv(output_csv, index=False)


def tm_align_substructures(substructures_dir:str,aligned_pdbs_dir:str,pdb_chain_table_path:str
                           ,structures_with_ubiqs_dir:str,
                           structures_without_ubiqs_dir:str):
    for file in os.listdir(substructures_dir):
        target_pdb = os.path.join(substructures_dir, file)
        # make sure the file is a pdb file
        if not target_pdb.endswith('.pdb'):
            continue
        align_uniprots_with_target(target_pdb=target_pdb, pdb_chain_table_path=pdb_chain_table_path, output_dir_path=aligned_pdbs_dir,
                                   structures_with_ubiqs_dir=structures_with_ubiqs_dir, structures_without_ubiqs_dir=structures_without_ubiqs_dir)

def add_structure_info(input_folder:str, input_csv:str, output_csv:str, patch_number:int):
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
        structure_csv = os.path.join(sub_folder, f"{uniprot}_top_scores.csv")
        
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
    plan_dict = {
        'version':'v4',
        'folds_training_dicts_path':'/home/iscb/wolfson/omriyakir/ubinet/datasets/patch_to_score/data_for_training/21_07_with_pesto/folds_training_dicts.pkl',
        'all_predictions_dict' : '/home/iscb/wolfson/doririmon/home/order/ubinet/repo/ubinet/results/ScanNet/all_predictions_2107_MSA_True.pkl',
        'model_results_dir_path': '/home/iscb/wolfson/omriyakir/ubinet/results/patch_to_score/hypotheses/pts_shalllow_mlps_autoregressive_scannet/21_07/2025-08-08_22693c7950',
        'protein_objs_path' : '/home/iscb/wolfson/omriyakir/ubinet/datasets/patch_to_score/patches_dicts/21_07_with_pesto/merged_protein_objects.pkl',
        'patch_number': 0,
        'create_ub_ratio': False,
        'create_sample_folds_training_dicts': False,
        'sample':False,
        'create_significances': False,
        'create_table':False,
        'create_top_k_by_source':False,
        'fetch_uniprots_data_from_api':False,
        'create_substructures': False,
        'tm_align_substructures': False,
        'add_structure_info': True,
        'add_uniprots_data': True,
        
    }
    candidates_dir = os.path.join(plan_dict['model_results_dir_path'], 'candidates')
    os.makedirs(candidates_dir, exist_ok=True)
    ub_ratios_dir = os.path.join(candidates_dir, 'ub_ratios')
    os.makedirs(ub_ratios_dir, exist_ok=True)
    substructures_dir = os.path.join(plan_dict['model_results_dir_path'], 'substructures_chainsaw',f'patch_{plan_dict["patch_number"]}')
    sample_folds_training_dicts_path = os.path.join(candidates_dir, 'sample_folds_training_dicts.pkl')
    significances_pkl_path = os.path.join(candidates_dir, f'significances_and_str_patches{"_sample" if plan_dict["sample"] else ""}.pkl')
    significances_csv_path = os.path.join(candidates_dir, f'significances_table{"_sample" if plan_dict["sample"] else ""}.csv')
    uniprots_data_csv_path = os.path.join(candidates_dir, 'uniprot_data_api.csv')
    sources = ['Arabidopsis proteome', 'Celegans proteome', 'Ecoli proteome', 'Human proteome', 'Yeast proteome']
    structures_with_ubiqs_dir = os.path.join(paths.binding_chains_pdbs_with_ubiqs_path, plan_dict['version'])
    structures_without_ubiqs_dir = os.path.join(paths.binding_chains_pdbs_without_ubiqs_path, plan_dict['version'])
    aligment_candidates_dir = os.path.join(candidates_dir,'aligned_candidates')
    
    if plan_dict['create_ub_ratio']:
        print('Creating ub ratio...')
        psu.create_training_ub_ratio(folds_training_dicts_path=plan_dict['folds_training_dicts_path'], ub_ratios_dir=ub_ratios_dir)

    if plan_dict['create_sample_folds_training_dicts']:
        print('Creating sample folds training dicts...')
        folds_training_dicts = load_as_pickle(plan_dict['folds_training_dicts_path'])
        sample_folds_training_dicts = []
        for i in range(len(folds_training_dicts)):
            fold = folds_training_dicts[i]
            sample_fold = {key:val[:100] for key,val in fold.items()}
            sample_folds_training_dicts.append(sample_fold)
        save_as_pickle(sample_folds_training_dicts, sample_folds_training_dicts_path)

    if plan_dict['create_significances']:
        print('Creating significances and str patches...')
        data_dict = psu.create_str_patches_and_Significances(
            folds_training_dicts_path='/home/iscb/wolfson/omriyakir/ubinet/datasets/patch_to_score/data_for_training/21_07_with_pesto/folds_training_dicts.pkl',
            model_dir=plan_dict['model_results_dir_path'],
            protein_locations_path=plan_dict['protein_objs_path'],
            ub_ratios_dir=ub_ratios_dir)
        save_as_pickle(data_dict, significances_pkl_path)
    
    if plan_dict['create_table']:
        print('Creating final table with sources and significance ll...')
        psu.create_csv_add_sources_and_significance_ll(
            data_dict_path=significances_pkl_path,
            proteins_path=plan_dict['protein_objs_path'],
            output_csv=significances_csv_path
        )
    
    if plan_dict['create_top_k_by_source']:
        print('Creating top k by source...')
        for source in sources:
            psu.filter_folds_training_dicts_by_source(
                source=source,
                input_csv_path=significances_csv_path,
                top_k=500,
                output_path = os.path.join(candidates_dir,f"significance_table_top_500_{source}.csv"),
            )

    if plan_dict['fetch_uniprots_data_from_api']:
        print('Fetching uniprot data from api...')
        df = pd.read_csv(significances_csv_path)
        uniprot_ids = df['uniprot'].unique().tolist()
        uniprot_to_csv(uniprot_ids=uniprot_ids, csv_path=uniprots_data_csv_path)


    if plan_dict['create_substructures']:
        print('Creating substructures for filtered examples...')
        all_predictions_ubiq = load_as_pickle(plan_dict['all_predictions_dict'])['dict_predictions_ubiquitin']
        for source in sources:
            input_csv = os.path.join(candidates_dir,f"significance_table_top_500_{source}.csv")
            output_path = os.path.join(candidates_dir, f'data_predictions_significances_filtered_{source}.csv')
            
            psu.create_substructures_for_filtered(
                csv_file=input_csv,
                substructures_dir=substructures_dir,
                all_predictions_ubiq=all_predictions_ubiq,
                patch_number=plan_dict['patch_number'],
                source=source,
                output_path=output_path

            )
        


    if plan_dict['tm_align_substructures']:
        print('TM-aligning substructures...')
        os.makedirs(aligment_candidates_dir, exist_ok=True)
        alignment_candidates_patchs_dir = os.path.join(aligment_candidates_dir,f'patch_{plan_dict["patch_number"]}')
        os.makedirs(alignment_candidates_patchs_dir, exist_ok=True)
        for source in sources:
            print(f'Processing source: {source}')
            substructures_dir_source = os.path.join(substructures_dir,source)
            os.makedirs(substructures_dir_source, exist_ok=True)
            alignment_candidates_patchs_source_dir = os.path.join(alignment_candidates_patchs_dir, source)
            os.makedirs(alignment_candidates_patchs_source_dir, exist_ok=True)

            pdb_chain_table_path = os.path.join(paths.structural_aligners_path,'tables',plan_dict['version'],'pdb_chain_uniprot_with_classes_table.csv')
            tm_align_substructures(substructures_dir=substructures_dir_source, aligned_pdbs_dir=alignment_candidates_patchs_source_dir, pdb_chain_table_path=pdb_chain_table_path,
                                structures_with_ubiqs_dir=structures_with_ubiqs_dir, structures_without_ubiqs_dir=structures_without_ubiqs_dir
                                )

    if plan_dict['add_structure_info']:
        print('Adding structure info to final table...')
        patch_number = plan_dict['patch_number']
        alignment_candidates_patchs_dir = os.path.join(aligment_candidates_dir,f'patch_{plan_dict["patch_number"]}')        
        for source in sources:
            print(f'Processing source: {source}')
            substructures_dir_source = os.path.join(substructures_dir,source)
            alignment_candidates_patchs_source_dir = os.path.join(alignment_candidates_patchs_dir, source)
            input_csv = os.path.join(candidates_dir, f'data_predictions_significances_filtered_{source}.csv')
            output_csv = os.path.join(candidates_dir,f"significance_table_top_500_{source}_aligments.csv")
            add_structure_info(input_folder=alignment_candidates_patchs_source_dir, input_csv=input_csv,
                                output_csv=output_csv, patch_number=patch_number)
    
    # ...existing code...
    if plan_dict['add_uniprots_data']:
        print('Adding uniprot data to final table...')
        for source in sources:
            print(f'Processing source: {source}')
            input_csv = os.path.join(candidates_dir, f"significance_table_top_500_{source}_aligments.csv")
            output_csv = os.path.join(candidates_dir, f"significance_table_top_500_{source}_final.csv")
            df = pd.read_csv(input_csv)
            uniprot_df = pd.read_csv(uniprots_data_csv_path)
            merged_df = pd.merge(df, uniprot_df, on='uniprot', how='left')
            
            # Reorder columns so that uniprot, Organism and Protein Name come first
            cols = merged_df.columns.tolist()
            desired_cols = ['uniprot', 'Organism', 'Protein Name']
            remaining_cols = [col for col in cols if col not in desired_cols]
            merged_df = merged_df[desired_cols + remaining_cols]
            
            # Round all numeric columns to 4 decimal places
            merged_df = merged_df.round(4)
            
            merged_df.to_csv(output_csv, index=False)


