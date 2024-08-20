import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from data_preparation.ScanNet.create_tables_and_weights import cluster_sequences
import numpy as np


# seqid=0.5, coverage=0.4


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



# def create_x_y_groups(allPredictionsPath, dirPath):
#     allPredictions = loadPickle(os.path.join(path.ScanNetPredictionsPath, allPredictionsPath))
#     trainingDictsDir = os.path.join(dirPath, 'trainingDicts')
#     allInfoDict = loadPickle(os.path.join(trainingDictsDir, 'allInfoDict.pkl'))
#     dictForTraining = loadPickle(os.path.join(trainingDictsDir, 'dictForTraining.pkl'))
#     allProteinsDict = dict()
#     allProteinsDict['x'] = allInfoDict['x_train'] + allInfoDict['x_cv'] + allInfoDict['x_test']
#     allProteinsDict['y'] = np.concatenate((allInfoDict['y_train'], allInfoDict['y_cv'], allInfoDict['y_test']))
#     components = np.concatenate((dictForTraining['x_train_components_scaled_padded'],
#                                  dictForTraining['x_cv_components_scaled_padded'],
#                                  dictForTraining['x_test_components_scaled_padded']), axis=0)
#     sizes = np.concatenate((dictForTraining['x_train_sizes_scaled'],
#                             dictForTraining['x_cv_sizes_scaled'],
#                             dictForTraining['x_test_sizes_scaled']), axis=0)
#     n_patches = np.concatenate((dictForTraining['x_train_n_patches_encoded'],
#                                 dictForTraining['x_cv_n_patches_encoded'],
#                                 dictForTraining['x_test_n_patches_encoded']), axis=0)
#
#     uniprots = [info[1] for info in allProteinsDict['x']]
#     sequences = [allPredictions['dict_sequences'][uniprot] for uniprot in uniprots]
#     cluster_indices, representative_indices = cluster_sequences(sequences)
#     clustersParticipantsList = createClusterParticipantsIndexes(cluster_indices)
#     clusterSizes = [l.size for l in clustersParticipantsList]
#     clusterSizesAndInedxes = [(i, clusterSizes[i]) for i in range(len(clusterSizes))]
#     sublists, sublistsSum = divideClusters(clusterSizesAndInedxes)
#     groupsIndexes = []
#
#     for l in sublists:
#         groupsIndexes.append(np.concatenate([clustersParticipantsList[index] for index in l]))
#
#     y_groups = []
#     x_groups = []
#     componentsGroups = []
#     sizesGroups = []
#     n_patchesGroups = []
#     assert components.shape[0] == sizes.shape[0] == n_patches.shape[0]
#     for indexGroup in groupsIndexes:
#         x = [allProteinsDict['x'][index] for index in indexGroup]
#         y = allProteinsDict['y'][indexGroup]
#         componentsGroup = components[indexGroup]
#         sizesGroup = sizes[indexGroup]
#         n_patchesGroup = n_patches[indexGroup]
#         x_groups.append(x)
#         y_groups.append(y)
#         componentsGroups.append(componentsGroup)
#         sizesGroups.append(sizesGroup)
#         n_patchesGroups.append(n_patchesGroup)
#
#     saveAsPickle(x_groups, os.path.join(trainingDictsDir, 'x_groups'))
#     saveAsPickle(y_groups, os.path.join(trainingDictsDir, 'y_groups'))
#     saveAsPickle(componentsGroups, os.path.join(trainingDictsDir, 'componentsGroups'))
#     saveAsPickle(sizesGroups, os.path.join(trainingDictsDir, 'sizesGroups'))
#     saveAsPickle(n_patchesGroups, os.path.join(trainingDictsDir, 'n_patchesGroups'))
#     return x_groups, y_groups, componentsGroups, sizesGroups, n_patchesGroups
