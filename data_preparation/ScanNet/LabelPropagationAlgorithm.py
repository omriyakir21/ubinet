import pdb
import sys
import os
import pdb
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import paths
import json

from data_preparation.ScanNet.db_creation_scanNet_utils import load_as_pickle,save_as_pickle
import LabelPropagationAlgorithm_utils as utils
from create_tables_and_weights import cluster_sequences

if __name__ == "__main__":
    plan_dict = {'name':'v4',
                 'seq_id': 0.95,
                 'ASA_THRESHOLD_VALUE': 0.1,
                 'NUM_PARTS': 100,
                 'divide_scanNet_to_parts':False,
                 'create_scanNet_data_in_parts':False,
                 'unite_scanNet_parts':False,
                 'normalize_asa':False,
                 'remove_non_valid_chains':False,
                 'debug':True,
                 'create_debug':True,
                 'propagate_duplicates':True,
                 'run_algorithm':True,
                 'divide_to_different_files':True}
    
    ImerFiles_path = os.path.join(paths.ImerFiles_path, plan_dict['name'])
    ASA_path = os.path.join(paths.ASA_path, plan_dict['name'])
    os.makedirs(ASA_path, exist_ok=True)
    mmseqs_scanNet_path = os.path.join(paths.mmseqs_scanNet_path, plan_dict['name'])
    os.makedirs(mmseqs_scanNet_path, exist_ok=True)
    mafft_path = os.path.join(paths.mafft_path, plan_dict['name'])
    os.makedirs(mafft_path, exist_ok=True)
    PSSM_path = os.path.join(paths.PSSM_path, plan_dict['name'])
    os.makedirs(PSSM_path, exist_ok=True)
    helper_path = os.path.join(paths.data_preperation_helper_path, plan_dict['name'])
    os.makedirs(helper_path, exist_ok=True)
    pdbs_not_included_path = os.path.join(helper_path, 'pdbs_not_included.txt')

    quantile_amino_acids_dict_path = os.path.join(helper_path, 'quantile_amino_acids_dict.pkl')
    imer_file_path = os.path.join(ImerFiles_path, 'Integrated_Checkchains_mer_filtered_converted.txt')
    asa_file_path = os.path.join(ASA_path, 'Integrated_Checkchains_asa_mer_filtered.txt')
    normalized_asa_file_path = os.path.join(ASA_path, 'normalizedFullASAPssmContent.txt')
    normalized_scanNet_asa_file_path = os.path.join(helper_path, 'normalized_asa_scanNet.txt')
    asa_scanNet_path = os.path.join(helper_path, 'asa_scanNet.txt')
    pssm_scanNet_path = os.path.join(helper_path, 'pssm_scanNet.txt')
    pssm_scanNet_removed_non_valid_chains_path = os.path.join(helper_path, 'pssm_scanNet_removed_non_valid_chains.txt')
    asa_scanNet_removed_non_valid_chains_path = os.path.join(helper_path, 'asa_scanNet_removed_non_valid_chains.txt')

    scanNet_labels_path = os.path.join(paths.ScanNet_Ub_PPBS_datasets, 'labels_aggregated_multiclass.txt')
    parts_folder = os.path.join(helper_path, 'parts')
    os.makedirs(parts_folder, exist_ok=True)
    input_parts_folder = os.path.join(parts_folder, 'input_parts')
    os.makedirs(input_parts_folder, exist_ok=True)
    scanNet_pssm_parts_folder = os.path.join(parts_folder, 'pssm')
    os.makedirs(scanNet_pssm_parts_folder, exist_ok=True)
    scanNet_asa_parts_folder = os.path.join(parts_folder, 'asa')
    os.makedirs(scanNet_asa_parts_folder, exist_ok=True)
    scanNet_not_included_parts_folder = os.path.join(parts_folder, 'not_included')
    os.makedirs(scanNet_not_included_parts_folder, exist_ok=True)

    if plan_dict['divide_scanNet_to_parts']:
        utils.divide_pssm_to_N_parts(folder=input_parts_folder,pssm_file=scanNet_labels_path, num_parts=plan_dict['NUM_PARTS'],)

    if plan_dict['create_scanNet_data_in_parts']:
        json_path = '/home/iscb/wolfson/omriyakir/other_works/PeSTo/labels_per_chain/v2/aggregated_header_to_filepath.json'
        pdb_dict = json.load(open(json_path, 'r'))
        part_index = int(sys.argv[1])

        utils.create_asa_file(input_parts_folder=input_parts_folder,scanNet_pssm_parts_folder=scanNet_pssm_parts_folder,
                    scanNet_asa_parts_folder=scanNet_asa_parts_folder,scanNet_not_included_parts_folder=scanNet_not_included_parts_folder,
                    index=part_index,pdb_dict=pdb_dict)
    
    if plan_dict['unite_scanNet_parts']:
        utils.unite_parts(scanNet_pssm_parts_folder=scanNet_pssm_parts_folder,scanNet_asa_parts_folder=scanNet_asa_parts_folder,
                scanNet_not_included_parts_folder=scanNet_not_included_parts_folder,num_parts=plan_dict['NUM_PARTS'],pssm_output_path = pssm_scanNet_path,
                asa_output_path = asa_scanNet_path,not_included_output_path=pdbs_not_included_path)


    if plan_dict['normalize_asa']:
        utils.normalize_asa_data(asa_scanNet_path,
                                    normalized_scanNet_asa_file_path, quantile_amino_acids_dict_path)

        utils.normalize_asa_data(asa_file_path,
                                normalized_asa_file_path,quantile_amino_acids_dict_path)

    if plan_dict['remove_non_valid_chains']:
        utils.remove_non_valid_chains(pssm_file=pssm_scanNet_path, asa_file=normalized_scanNet_asa_file_path,
                                      output_pssm_file=pssm_scanNet_removed_non_valid_chains_path,
                                      output_asa_file=asa_scanNet_removed_non_valid_chains_path)


    if plan_dict['create_debug']:

        utils.sample_pssm_and_asa_files(input_pssm =pssm_scanNet_removed_non_valid_chains_path,input_asa=asa_scanNet_removed_non_valid_chains_path,
                                         output_pssm_debug= os.path.join(helper_path, 'pssm_scanNet_removed_non_valid_chains_debug.txt'),
                                        output_asa_debug=os.path.join(helper_path, 'asa_scanNet_removed_non_valid_chains_debug.txt'),
                                        sample_n=700, seed=42)
    
    debug_addition = '_debug' if plan_dict['debug'] else ''
    normalized_scanNet_asa_file_path = os.path.join(helper_path, f'asa_scanNet_removed_non_valid_chains{debug_addition}.txt')
    pssm_scanNet_path = os.path.join(helper_path, f'pssm_scanNet_removed_non_valid_chains{debug_addition}.txt')
    pssm_after_duplicates_processing_path = os.path.join(helper_path, f'ubiq_pssm_after_duplicates_processing{debug_addition}.txt')
    scanNet_pssm_after_duplicates_processing_path = os.path.join(helper_path, f'scanNet_pssm_after_duplicates_processing{debug_addition}.txt')
    scanNet_asa_after_duplicates_processing_path = os.path.join(helper_path, f'scanNet_asa_after_duplicates_processing{debug_addition}.txt')
    ubiq_and_scanNet_pssm_after_duplication_processing_path = os.path.join(helper_path, f'ubiq_and_scanNet_pssm_after_duplication_processing{debug_addition}.txt')
    ubiq_and_scanNet_asa_after_duplication_processing_path = os.path.join(helper_path, f'ubiq_and_scanNet_asa_after_duplication_processing{debug_addition}.txt')
    ubiq_and_scanNet_pssm_final = os.path.join(helper_path, f'ubiq_and_scanNet_pssm_final{debug_addition}.txt')
    ubiq_and_scanNet_asa_final = os.path.join(helper_path, f'ubiq_and_scanNet_asa_final{debug_addition}.txt')

    if plan_dict['propagate_duplicates']:
        ubiq_chain_dict, scanNet_chain_dict, scanNet_chain_dict_asa = utils.propagate_labels_and_concat(ubiq_PSSM=imer_file_path,
                                    scanNet_PSSM=pssm_scanNet_path,normalized_asa_scanNet_file=normalized_scanNet_asa_file_path,
                                    ubiq_output_PSSM=pssm_after_duplicates_processing_path, scanNet_output_PSSM=scanNet_pssm_after_duplicates_processing_path,
                                    scanNet_output_asa=scanNet_asa_after_duplicates_processing_path)

        utils.concat_files_with_newline_seperator([pssm_after_duplicates_processing_path, scanNet_pssm_after_duplicates_processing_path],
                        ubiq_and_scanNet_pssm_after_duplication_processing_path)

        utils.concat_files_with_newline_seperator([normalized_asa_file_path, scanNet_asa_after_duplicates_processing_path],
                        ubiq_and_scanNet_asa_after_duplication_processing_path)

        utils.propagate_for_duplicate_sequences(PSSM_file=ubiq_and_scanNet_pssm_after_duplication_processing_path,
                                                asa_file=ubiq_and_scanNet_asa_after_duplication_processing_path,
                                                output_PSSM_file=ubiq_and_scanNet_pssm_final,
                                                output_asa_file=ubiq_and_scanNet_asa_final)
        
    helper_algorithm_folder = os.path.join(helper_path, 'helper_algorithm')
    os.makedirs(helper_algorithm_folder, exist_ok=True)
    
    splitted_data_path = os.path.join(helper_algorithm_folder, f'splitted_data{debug_addition}.pkl')
    cluster_indices_path = os.path.join(helper_algorithm_folder, f'clusterIndices_{plan_dict["seq_id"]}{debug_addition}.pkl')
    clusters_participants_list_path = os.path.join(helper_algorithm_folder, f'clustersParticipantsList_{plan_dict["seq_id"]}{debug_addition}.pkl')
    clusters_dict_path = os.path.join(helper_algorithm_folder, f'clustersDict_{plan_dict["seq_id"]}{debug_addition}.pkl')
    PSSM_seq_id_folder = os.path.join(PSSM_path,f'seq_id_{plan_dict["seq_id"]}_asaThreshold_{plan_dict["ASA_THRESHOLD_VALUE"]}')
    os.makedirs(PSSM_seq_id_folder, exist_ok=True)
    propagated_pssm_file_path = os.path.join(PSSM_seq_id_folder, f'propagatedPssmFile_{plan_dict["seq_id"]}_asaThreshold_{plan_dict["ASA_THRESHOLD_VALUE"]}{debug_addition}.txt')

    if plan_dict['run_algorithm']:

        ubiq_chains_keys, ubiq_chains_sequences, ubiq_chains_labels, ubiq_chain_names, ubiq_lines, ubiq_chains_asa_values = utils.split_receptors_into_individual_chains(
        pssm_after_duplicates_processing_path,
        normalized_asa_file_path)

        scanNet_chains_keys, scanNet_chains_sequences, scanNet_chains_labels, scanNet_chain_names, scanNet_lines, scanNet_chains_asa_values = utils.split_receptors_into_individual_chains(
        scanNet_pssm_after_duplicates_processing_path,scanNet_asa_after_duplicates_processing_path)

        chains_keys, chains_sequences, chains_labels, chains_names, lines, chains_asa_values = utils.split_receptors_into_individual_chains(
        ubiq_and_scanNet_pssm_final,ubiq_and_scanNet_asa_final)

        splited_data = {'chains_keys': chains_keys, 'chains_sequences': chains_sequences, 'chains_labels': chains_labels,
                        'chain_names': chains_names, 'lines': lines, 'chains_asa_values': chains_asa_values}
        # splited_data = load_as_pickle(splitted_data_path)

        # # #
        cluster_indices, representative_indices = cluster_sequences(chains_sequences,
                                                                    seqid=plan_dict['seq_id'],
                                                                    coverage=0.9, covmode='0',
                                                                    path2mmseqstmp=paths.tmp_path,
                                                                    path2mmseqs=paths.mmseqs_exec_path)
        save_as_pickle(chains_sequences, os.path.join(helper_algorithm_folder, 'chains_sequences.pkl'))
        save_as_pickle(cluster_indices, cluster_indices_path)
        # cluster_indices = load_as_pickle(cluster_indices_path)
        clusters_participants_list = utils.create_cluster_participants_indexes(cluster_indices)
        save_as_pickle(clusters_participants_list,clusters_participants_list_path)
        # clusters_participants_list = load_as_pickle(clusters_participants_list_path)
        clusters_dict = utils.apply_mafft_for_all_clusters(chains_sequences, clusters_participants_list,
                                                        paths.mafft_exec_path)
        save_as_pickle(clusters_dict, clusters_dict_path)
        # clusters_participants_list = load_as_pickle(clusters_participants_list_path)
        # clusters_dict = load_as_pickle(clusters_dict_path)
        utils.create_propagated_pssm_file(
            clusters_dict,
            splited_data['chains_labels'],
            clusters_participants_list,
            splited_data['chains_sequences'],
            splited_data['chain_names'],
            splited_data['lines'],
            chains_asa_values,
            propagated_pssm_file_path,
            plan_dict['ASA_THRESHOLD_VALUE']
        )

        
    if plan_dict['divide_to_different_files']:       
        propagated_pssm_only_ubiq_file_path = os.path.join(PSSM_seq_id_folder, f'propagatedPssmFile_only_ubiq_{plan_dict["seq_id"]}_asaThreshold_{plan_dict["ASA_THRESHOLD_VALUE"]}{debug_addition}.txt')
        propagated_pssm_only_scanNet_file_path = os.path.join(PSSM_seq_id_folder, f'propagatedPssmFile_only_scanNet_{plan_dict["seq_id"]}_asaThreshold_{plan_dict["ASA_THRESHOLD_VALUE"]}{debug_addition}.txt')
        propagated_pssm_ubiqs_with_added_scanNet_homologs_file_path = os.path.join(PSSM_seq_id_folder, f'propagatedPssmFile_ubiqs_with_scanNet_homologs{plan_dict["seq_id"]}_asaThreshold_{plan_dict["ASA_THRESHOLD_VALUE"]}{debug_addition}.txt')
        propagated_pssm_scanNet_non_ubiq_homologs_file_path = os.path.join(PSSM_seq_id_folder, f'propagatedPssmFile_scanNet_non_ubiq_homologs{plan_dict["seq_id"]}_asaThreshold_{plan_dict["ASA_THRESHOLD_VALUE"]}{debug_addition}.txt')
        
        ubiq_chains_keys, _, ubiq_chains_labels, _, _, _ = utils.split_receptors_into_individual_chains(
        pssm_after_duplicates_processing_path,
        normalized_asa_file_path)


        scanNet_chains_keys, _, _, _, _, _ = utils.split_receptors_into_individual_chains(
        scanNet_pssm_after_duplicates_processing_path,scanNet_asa_after_duplicates_processing_path)
        
        propagated_chains_keys, _, propagated_chains_labels, propagated_chain_names, _, _ = utils.split_receptors_into_individual_chains(
            propagated_pssm_file_path,ubiq_and_scanNet_asa_final)

        utils.filter_pssm_using_keys(propagated_pssm_file_path,ubiq_chains_keys,propagated_pssm_only_ubiq_file_path)
        utils.filter_pssm_using_keys(propagated_pssm_file_path,scanNet_chains_keys,propagated_pssm_only_scanNet_file_path)
        
        set_ubiqs_keys = set(ubiq_chains_keys)
        binders_keys = []        
        for i in range(len(propagated_chain_names)):
            key = propagated_chain_names[i].split("$")[0]
            if key in set_ubiqs_keys or '2' in propagated_chains_labels[i]:
                binders_keys.append(key)
        
        utils.filter_pssm_using_keys(propagated_pssm_file_path,binders_keys,propagated_pssm_ubiqs_with_added_scanNet_homologs_file_path)
        
        binders_keys_set = set(binders_keys)
        scanNet_non_binding_keys = []
        for key in propagated_chains_keys:
            if key not in binders_keys_set:
                scanNet_non_binding_keys.append(key)
        
        utils.filter_pssm_using_keys(propagated_pssm_file_path,scanNet_non_binding_keys,propagated_pssm_scanNet_non_ubiq_homologs_file_path)