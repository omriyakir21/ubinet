import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import cath_utils
import paths
from data_preparation.ScanNet.db_creation_scanNet_utils import save_as_pickle

if __name__ == "__main__":
    dataset_date = "8_9"
    seq_id = "0.95"
    ASA_THRESHOLD_VALUE = 0.2
    with_scanNet = True
    with_scanNet_addition = '_with_scanNet_' if with_scanNet else ''
    os.makedirs(paths.cath_intermediate_files_path, exist_ok=True)
    
    pssm_folder = os.path.join(paths.PSSM_path,f'PSSM_{dataset_date}', f'seq_id_{seq_id}_asaThreshold_{ASA_THRESHOLD_VALUE}')
    full_pssm_file_path = os.path.join(os.path.join(pssm_folder, f'propagatedPssmWithAsaFile_{seq_id}_asaThreshold_{ASA_THRESHOLD_VALUE}.txt'))
    cath_df = cath_utils.make_cath_df_new(os.path.join(paths.cath_path, "cath_b.20230204.txt"))

    if with_scanNet:
        scanNet_PSSM = cath_utils.scanNet_PSSM_files_concatenation()
        ubiq_output_PSSM, scanNet_output_PSSM = cath_utils.propagate_labels_and_concat(full_pssm_file_path,scanNet_PSSM,pssm_folder)
        full_pssm_file_path = f'{full_pssm_file_path.split(".txt")[0]}{with_scanNet_addition}.txt'
        cath_utils.concat_and_remove_duplicates(ubiq_output_PSSM, scanNet_output_PSSM, full_pssm_file_path)

    names_list, sizes_list, sequence_list, full_names_list, pdb_names_with_chains_lists = cath_utils.list_creation(
        full_pssm_file_path)

    
    structuresDicts = cath_utils.create_dictionaries(names_list, sizes_list, sequence_list, full_names_list,
                                                     pdb_names_with_chains_lists)
    inCath, notInCath, cnt = cath_utils.count_in_cath(cath_df, structuresDicts)
    cath_utils.find_chains_in_cath(cath_df, structuresDicts)
    cath_utils.add_classifications_for_dict(cath_df, structuresDicts, 4)
    
    matHomologous_path = os.path.join(paths.cath_intermediate_files_path, f'matHomologous_{dataset_date}{with_scanNet_addition}.pkl')
    graphHomologous_path = os.path.join(paths.cath_intermediate_files_path, f'graphHomologous_{dataset_date}{with_scanNet_addition}.pkl')
    if os.path.exists(matHomologous_path) and os.path.exists(graphHomologous_path):
        matHomologous = cath_utils.load_from_pickle(matHomologous_path)
        graphHomologous = cath_utils.load_from_pickle(graphHomologous_path)
    else:
        matHomologous = cath_utils.neighbor_mat_new(structuresDicts)
        graphHomologous = cath_utils.csr_matrix(matHomologous)
        save_as_pickle(matHomologous,matHomologous_path)
        save_as_pickle(graphHomologous,graphHomologous_path)
    
    homologous_components, homologousLabels = cath_utils.connected_components(csgraph=graphHomologous, directed=False,
                                                                              return_labels=True)
    chainDict = cath_utils.components_to_chain_dict(homologous_components, homologousLabels,sizes_list,full_names_list)
    cath_utils.divide_pssm(chainDict, full_pssm_file_path,pssm_folder,with_scanNet_addition)
