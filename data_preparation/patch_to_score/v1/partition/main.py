import os
from typing import List
import numpy as np
import tensorflow as tf
from utils import load_as_tensor, create_paths
from data_preparation.ScanNet.db_creation_scanNet_utils import save_as_pickle, load_as_pickle
from data_preparation.ScanNet.create_tables_and_weights import cluster_sequences


def create_training_folds(groups_indices: List[np.array], folds_amount: int, 
                          scaled_sizes_path: str, scaled_components_list_path: str, encoded_components_list_path: str,
                          all_uniprots_path: str, labels_path: str):
    folds_training_dicts = []
    scaled_sizes = load_as_tensor(scaled_sizes_path)
    scaled_components = load_as_tensor(scaled_components_list_path)
    encoded_components = load_as_tensor(encoded_components_list_path)
    labels = load_as_tensor(labels_path, tf.int32)
    uniprots = load_as_pickle(all_uniprots_path)

    for i in range(folds_amount):
        training_dict = {}

        training_indices = tf.constant(np.concatenate([groups_indices[(i + j) % folds_amount] for j in range(2, folds_amount)]))
        validation_indices = tf.constant(groups_indices[i])
        test_indices = tf.constant(groups_indices[(i + 1) % folds_amount])

        # When using these indices to index tensors
        training_dict['sizes_train'] = tf.gather(
            scaled_sizes, training_indices)
        training_dict['components_train'] = tf.gather(
            scaled_components, training_indices)
        training_dict['num_patches_train'] = tf.gather(
            encoded_components, training_indices)
        # Assuming uniprots is a list and not a tensor
        training_dict['uniprots_train'] = [uniprots[i]
                                           for i in training_indices.numpy()]
        training_dict['labels_train'] = tf.gather(labels, training_indices)
        training_dict['sizes_validation'] = tf.gather(
            scaled_sizes, validation_indices)
        training_dict['components_validation'] = tf.gather(
            scaled_components, validation_indices)
        training_dict['num_patches_validation'] = tf.gather(
            encoded_components, validation_indices)
        # Assuming uniprots is a list and not a tensor
        training_dict['uniprots_validation'] = [uniprots[i]
                                                for i in validation_indices.numpy()]
        training_dict['labels_validation'] = tf.gather(
            labels, validation_indices)
        training_dict['sizes_test'] = tf.gather(scaled_sizes, test_indices)
        training_dict['components_test'] = tf.gather(
            scaled_components, test_indices)
        training_dict['num_patches_test'] = tf.gather(
            encoded_components, test_indices)
        # Assuming uniprots is a list and not a tensor
        training_dict['uniprots_test'] = [uniprots[i]
                                          for i in test_indices.numpy()]
        training_dict['labels_test'] = tf.gather(labels, test_indices)

        folds_training_dicts.append(training_dict)
    return folds_training_dicts


def create_cluster_participants_indices(cluster_indices):
    clustersParticipantsList = []
    for i in range(np.max(cluster_indices) + 1):
        clustersParticipantsList.append(np.where(cluster_indices == i)[0])
    return clustersParticipantsList


def divide_clusters(cluster_sizes, folds_amount: int):
    """
    :param cluster_sizes: list of tuples (clusterIndex,size)
    :return:  sublists,sublistsSum
    divide the list into <folds_amount> sublists such that the sum of each cluster sizes in the sublist is as close as possible
    """
    sublists = [[] for _ in range(folds_amount)]
    sublistsSum = [0 for _ in range(folds_amount)]
    # Sort the clusters by size descending order.
    cluster_sizes.sort(reverse=True, key=lambda x: x[1])
    for tup in cluster_sizes:
        # find the cluster with the minimal sum
        min_cluster_index = sublistsSum.index(min(sublistsSum))
        sublistsSum[min_cluster_index] += tup[1]
        sublists[min_cluster_index].append(tup[0])
    return sublists, sublistsSum


def get_uniprot_indices_for_groups(clusters_participants_list, sublists, fold_num: int) -> np.array:
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


def partition_to_folds_and_save(sequences: List[str], sequence_identity: float, coverage: float, folds_amount: int,
                                save_dir_path: str, path2mmseqs: str, path2mmseqstmp: str):
    cluster_indices, representative_indices = cluster_sequences(sequences, 
                                                                seqid=sequence_identity, 
                                                                coverage=coverage,
                                                                path2mmseqstmp=path2mmseqstmp,
                                                                path2mmseqs=path2mmseqs)
    save_as_pickle(cluster_indices, os.path.join(
        save_dir_path, 'cluster_indices.pkl'))
    clusters_participants_list = create_cluster_participants_indices(
        cluster_indices)
    cluster_sizes = [l.size for l in clusters_participants_list]
    cluster_sizes_and_indices = [(i, cluster_sizes[i])
                                 for i in range(len(cluster_sizes))]
    sublists, sublists_sum = divide_clusters(cluster_sizes_and_indices, folds_amount)
    groups_indices = [get_uniprot_indices_for_groups(clusters_participants_list, sublists, fold_num) for fold_num
                      in
                      range(folds_amount)]
    save_as_pickle(groups_indices, os.path.join(
        save_dir_path, 'groups_indices.pkl'))
    # CREATE TRAINING DICTS
    folds_training_dicts = create_training_folds(groups_indices, folds_amount,
                                                 os.path.join(save_dir_path,
                                                              'scaled_sizes.tf'),
                                                 os.path.join(save_dir_path,
                                                              'scaled_components_list.tf'),
                                                 os.path.join(save_dir_path,
                                                              'encoded_components_list.tf'),
                                                 os.path.join(
                                                     save_dir_path, 'uniprots.pkl'),
                                                 os.path.join(save_dir_path, 'labels.tf'))
    print(f'before saving folds dict')
    save_as_pickle(folds_training_dicts, os.path.join(save_dir_path,
                                                      'folds_training_dicts.pkl'))


def create_uniprots_sets(folds_amount: int, save_dir_path: str):
    uniprots = load_as_pickle(os.path.join(
        save_dir_path, 'uniprots.pkl'))
    groups_indices = load_as_pickle(os.path.join(
        save_dir_path, 'groups_indices.pkl'))
    uniprots_sets = []
    for i in range(folds_amount):
        uniprots_set = set([uniprots[j] for j in groups_indices[(i+1) % folds_amount]])
        uniprots_sets.append(uniprots_set)
    save_as_pickle(uniprots_sets, os.path.join(
        save_dir_path, 'uniprots_sets.pkl'))


def partition(sequences: List[str], sequence_identity: float, coverage: float, folds_amount: int,
              save_dir_path: str, path2mmseqs: str, path2mmseqstmp: str):
    create_paths(path2mmseqstmp)
    partition_to_folds_and_save(sequences, sequence_identity, coverage, folds_amount,
                                save_dir_path, path2mmseqs, path2mmseqstmp)
    create_uniprots_sets(folds_amount, save_dir_path)
