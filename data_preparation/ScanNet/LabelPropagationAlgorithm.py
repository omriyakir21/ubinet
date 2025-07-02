import pdb
import sys
import os
import pdb
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import paths
from data_preparation.ScanNet.db_creation_scanNet_utils import load_as_pickle,save_as_pickle
import LabelPropagationAlgorithm_utils as utils
from create_tables_and_weights import cluster_sequences

if __name__ == "__main__":
    name = 'v1'
    seq_id = 0.95
    ASA_THRESHOLD_VALUE = 0.2
    ImerFiles_path = os.path.join(paths.ImerFiles_path, name)
    ASA_path = os.path.join(paths.ASA_path, name)
    mmseqs_scanNet_path = os.path.join(paths.mmseqs_scanNet_path, name)
    os.makedirs(mmseqs_scanNet_path, exist_ok=True)
    mafft_path = os.path.join(paths.mafft_path, name)
    os.makedirs(mafft_path, exist_ok=True)
    PSSM_path = os.path.join(paths.PSSM_path, name)
    os.makedirs(PSSM_path, exist_ok=True)

    imer_file_path = os.path.join(ImerFiles_path, 'Integrated_Checkchains_mer_filtered.txt')
    asa_file_path = os.path.join(ASA_path, 'Integrated_Checkchains_asa_mer_filtered.txt')
    normalized_asa_file_path = os.path.join(ASA_path, 'normalizedFullASAPssmContent.txt')
    utils.normalize_asa_data(asa_file_path,
                             normalized_asa_file_path,ASA_path)
    chains_keys, chains_sequences, chains_labels, chain_names, lines, chains_asa_values = utils.split_receptors_into_individual_chains(
        imer_file_path,
        normalized_asa_file_path)
    splited_data = {'chains_keys': chains_keys, 'chains_sequences': chains_sequences, 'chains_labels': chains_labels,
                    'chain_names': chain_names, 'lines': lines, 'chains_asa_values': chains_asa_values}
    utils.save_as_pickle(splited_data, os.path.join(mmseqs_scanNet_path, 'splitedData.pkl'))
    splited_data = load_as_pickle(os.path.join(mmseqs_scanNet_path, 'splitedData.pkl'))
    chains_keys, chains_sequences, chains_labels, chain_names, lines, chains_asa_values = splited_data['chains_keys'], \
        splited_data['chains_sequences'], splited_data['chains_labels'], splited_data['chain_names'], splited_data[
        'lines'], \
        splited_data['chains_asa_values']
    # # #
    cluster_indices, representative_indices = cluster_sequences(chains_sequences,
                                                                seqid=seq_id,
                                                                coverage=0.8, covmode='0',
                                                                path2mmseqstmp=paths.tmp_path,
                                                                path2mmseqs=paths.mmseqs_exec_path)
    
    save_as_pickle(cluster_indices, os.path.join(mmseqs_scanNet_path, f'clusterIndices_{seq_id}.pkl'))
    cluster_indices = load_as_pickle(os.path.join(mmseqs_scanNet_path, f'clusterIndices_{seq_id}.pkl'))
    clusters_participants_list = utils.create_cluster_participants_indexes(cluster_indices)
    save_as_pickle(clusters_participants_list, os.path.join(mmseqs_scanNet_path, f'clustersParticipantsList_{seq_id}.pkl'))
    clusters_dict = utils.apply_mafft_for_all_clusters(chains_sequences, clusters_participants_list,
                                                       paths.mafft_exec_path)
    save_as_pickle(clusters_dict, os.path.join(mafft_path, f'clustersDict_{seq_id}.pkl'))
    clusters_participants_list = load_as_pickle(os.path.join(mmseqs_scanNet_path, f'clustersParticipantsList_{seq_id}.pkl'))
    clusters_dict = load_as_pickle(os.path.join(mafft_path, f'clustersDict_{seq_id}.pkl'))
    PSSM_seq_id_folder = os.path.join(PSSM_path,f'seq_id_{seq_id}_asaThreshold_{ASA_THRESHOLD_VALUE}')
    os.makedirs(PSSM_seq_id_folder, exist_ok=True)
    utils.create_propagated_pssm_file(clusters_dict, chains_labels, clusters_participants_list, chains_sequences,
                                      chain_names, lines,
                                      chains_asa_values, os.path.join(PSSM_seq_id_folder,f'propagatedPssmWithAsaFile_{seq_id}_asaThreshold_{ASA_THRESHOLD_VALUE}.txt'),
                                      ASA_THRESHOLD_VALUE)