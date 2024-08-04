import sys
import os
import paths

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(paths.current_dir, '..'))

import LabelPropagationAlgorithm_utils as utils

if __name__ == "__main__":
    path_to_mafft_exec = path2mafft = '/usr/bin/mafft'
    ubuntu = False
    if ubuntu:
        rootPath = '/mnt/c/Users/omriy/ubinet/'
    else:
        rootPath = paths.current_dir

    chains_keys, chains_sequences, chains_labels, chain_names, lines, chains_asa_values = utils.split_receptors_into_individual_chains(
        os.path.join(rootPath, 'FullPssmContent.txt'), os.path.join(rootPath, 'normalizedFullASAPssmContent.txt'))
    path2mmseqstmp = paths.mmseqs_scanNet_path
    cluster_indices, representative_indices = utils.cluster_sequences(chains_sequences, path2mmseqstmp, seqid=0.95,
                                                                      coverage=0.8, covmode='0')
    cluster_indexes = utils.load_as_pickle(os.path.join(paths.mmseqs_scanNet_path, 'clusterIndices.pkl'))
    clusters_participants_list = utils.create_cluster_participants_indexes(cluster_indexes)
    clusters_dict = utils.apply_mafft_for_all_clusters(chains_sequences, clusters_participants_list, path_to_mafft_exec)
    utils.save_as_pickle(clusters_dict, os.path.join(paths.mafft_path, 'clustersDict.pkl'))
    clusters_dict = utils.load_as_pickle(os.path.join(paths.mafft_path, 'clustersDict.pkl'))
    utils.create_propagated_pssm_file(clusters_dict, chains_labels, clusters_participants_list, chains_sequences,
                                      chain_names, lines,
                                      chains_asa_values)
    utils.normalize_asa_data(os.path.join(paths.PSSM_path, 'propagatedPssmWithAsaFile.txt'))
