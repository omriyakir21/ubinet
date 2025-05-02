import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from data_preparation.ScanNet.create_tables_and_weights import cluster_sequences
import numpy as np
from data_preparation.ScanNet.db_creation_scanNet_utils import save_as_pickle,load_as_pickle
from data_development_utils import create_training_folds
import paths


def create_cluster_participants_indices(cluster_indices):
    clustersParticipantsList = []
    for i in range(np.max(cluster_indices) + 1):
        clustersParticipantsList.append(np.where(cluster_indices == i)[0])
    return clustersParticipantsList


def divide_clusters(cluster_sizes):
    """
    :param cluster_sizes: list of tuples (clusterIndex,size)
    :return:  sublists,sublistsSum
    divide the list into 5 sublists such that the sum of each cluster sizes in the sublist is as close as possible
    """
    sublists = [[] for i in range(5)]
    sublistsSum = [0 for i in range(5)]
    cluster_sizes.sort(reverse=True, key=lambda x: x[1])  # Sort the clusters by size descending order.
    for tup in cluster_sizes:
        min_cluster_index = sublistsSum.index(min(sublistsSum))  # find the cluster with the minimal sum
        sublistsSum[min_cluster_index] += tup[1]
        sublists[min_cluster_index].append(tup[0])
    return sublists, sublistsSum


def get_uniprot_indices_for_groups(clusters_participants_list, sublists, fold_num):
    """
    Get the Uniprot indices for a specific fold.

    :param clusters_participants_list: List of np arrays where each array contains the indices of Uniprots in that fold.
    :param sublists: List of lists where each sublist contains the cluster indices for that fold.
    :param fold_num: The fold number for which to get the Uniprot indices.
    :return: List of Uniprot indices for the specified fold.
    """

    uniprot_indices = []
    for cluster_index in sublists[fold_num]:
        fold_indices = list(clusters_participants_list[cluster_index])
        uniprot_indices.append(fold_indices)
    return np.concatenate(uniprot_indices)

def partition_to_folds_and_save(sequences,data_for_training_folder_path):
    cluster_indices, representative_indices = cluster_sequences(sequences, seqid=0.5, coverage=0.4,
                                                                path2mmseqstmp=paths.tmp_path,
                                                                path2mmseqs=paths.mmseqs_exec_path)
    save_as_pickle(cluster_indices, os.path.join(data_for_training_folder_path, 'cluster_indices.pkl'))
    clusters_participants_list = create_cluster_participants_indices(cluster_indices)
    cluster_sizes = [l.size for l in clusters_participants_list]
    cluster_sizes_and_indices = [(i, cluster_sizes[i]) for i in range(len(cluster_sizes))]
    sublists, sublists_sum = divide_clusters(cluster_sizes_and_indices)
    groups_indices = [get_uniprot_indices_for_groups(clusters_participants_list, sublists, fold_num) for fold_num
                      in
                      range(5)] 
    save_as_pickle(groups_indices, os.path.join(data_for_training_folder_path, 'groups_indices.pkl'))
        # CREATE TRAINING DICTS
    folds_training_dicts = create_training_folds(groups_indices,
                                                           os.path.join(data_for_training_folder_path,
                                                                        'scaled_sizes.tf'),
                                                           os.path.join(data_for_training_folder_path,
                                                                        'scaled_components_list.tf'),
                                                           os.path.join(data_for_training_folder_path,
                                                                        'encoded_components_list.tf'),
                                                           os.path.join(data_for_training_folder_path, 'uniprots.pkl'),
                                                           os.path.join(data_for_training_folder_path, 'labels.tf'))
    print(f'before saving folds dict')
    save_as_pickle(folds_training_dicts,os.path.join(data_for_training_folder_path,
                                                                        'folds_training_dicts.pkl'))

def create_uniprots_sets(data_for_training_folder_path):
    uniprots = load_as_pickle(os.path.join(data_for_training_folder_path, 'uniprots.pkl'))
    groups_indices = load_as_pickle(os.path.join(data_for_training_folder_path, 'groups_indices.pkl'))
    uniprots_sets = []
    for i in range(5):
        uniprots_set = set([uniprots[j] for j in groups_indices[(i+1)%5]])
        uniprots_sets.append(uniprots_set)
    save_as_pickle(uniprots_sets, os.path.join(data_for_training_folder_path, 'uniprots_sets.pkl'))
