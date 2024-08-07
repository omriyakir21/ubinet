import pdb
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import paths

import LabelPropagationAlgorithm_utils as utils
from create_tables_and_weights import cluster_sequences

if __name__ == "__main__":
    # utils.normalize_asa_data(os.path.join(paths.ASA_path, 'Integrated_Checkchains_asa_mer.txt'),
    #                          os.path.join(paths.ASA_path, 'normalizedFullASAPssmContent.txt'))
    # chains_keys, chains_sequences, chains_labels, chain_names, lines, chains_asa_values = utils.split_receptors_into_individual_chains(
    #     os.path.join(paths.ImerFiles_path, 'Integrated_Checkchains_mer.txt'),
    #     os.path.join(os.path.join(paths.ASA_path, 'normalizedFullASAPssmContent.txt')))
    # splited_data = {'chains_keys': chains_keys, 'chains_sequences': chains_sequences, 'chains_labels': chains_labels,
    #                 'chain_names': chain_names, 'lines': lines, 'chains_asa_values': chains_asa_values}
    # utils.save_as_pickle(splited_data, os.path.join(paths.mmseqs_scanNet_path, 'splitedData.pkl'))
    splited_data = utils.load_as_pickle(os.path.join(paths.mmseqs_scanNet_path, 'splitedData.pkl'))
    chains_keys, chains_sequences, chains_labels, chain_names, lines, chains_asa_values = splited_data['chains_keys'], \
        splited_data['chains_sequences'], splited_data['chains_labels'], splited_data['chain_names'], splited_data[
        'lines'], \
        splited_data['chains_asa_values']
    #
    # cluster_indices, representative_indices = cluster_sequences(chains_sequences,
    #                                                             seqid=0.95,
    #                                                             coverage=0.8, covmode='0',
    #                                                             path2mmseqstmp=paths.tmp_path,
    #                                                             path2mmseqs=paths.mmseqs_exec_path)
    #
    # utils.save_as_pickle(cluster_indices, os.path.join(paths.mmseqs_scanNet_path, 'clusterIndices.pkl'))
    cluster_indices = utils.load_as_pickle(os.path.join(paths.mmseqs_scanNet_path, 'clusterIndices.pkl'))
    clusters_participants_list = utils.create_cluster_participants_indexes(cluster_indices)
    clusters_dict = utils.apply_mafft_for_all_clusters(chains_sequences, clusters_participants_list,
                                                       paths.mafft_exec_path)
    utils.save_as_pickle(clusters_dict, os.path.join(paths.mafft_path, 'clustersDict.pkl'))
    clusters_dict = utils.load_as_pickle(os.path.join(paths.mafft_path, 'clustersDict.pkl'))
    ASA_THRESHOLD_VALUE = 0.2
    utils.create_propagated_pssm_file(clusters_dict, chains_labels, clusters_participants_list, chains_sequences,
                                      chain_names, lines,
                                      chains_asa_values, os.path.join(paths.PSSM_path, 'propagatedPssmWithAsaFile.txt'),
                                      ASA_THRESHOLD_VALUE)
