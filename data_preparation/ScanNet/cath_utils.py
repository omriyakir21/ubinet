import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import os.path

import numpy as np
import pandas as pd
from Bio import pairwise2
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import pickle

import paths


def calculate_identity(seqA, seqB):
    """
    :param seqA: The sequence of amino acid from chain A
    :param seqB: The sequence of amino acid from chain B
    :return: percentage of identity between the sequences
    """
    score = pairwise2.align.globalxx(seqA, seqB, one_alignment_only=True, score_only=True)
    max_len = max(len(seqA), len(seqB))
    identity = score / max_len
    return identity


def keep_only_chars(string):
    return ''.join([char for char in string if char.isalpha()])


def list_creation_util(full_name):
    pdb_name = full_name[0:4]
    chains_string = full_name.split('_')[1]
    chains_strings_list = chains_string.split('+')
    chains_names_list = [keep_only_chars(chainString) for chainString in chains_strings_list]
    pdb_names_with_chains_list = [pdb_name + chainName for chainName in chains_names_list]
    return pdb_name, pdb_names_with_chains_list


def create_dictionaries(names_list, sizes_list, sequence_lists, full_names_list, pdb_names_with_chains_lists):
    structures_dicts = {}
    for i in range(len(full_names_list)):
        structureDict = {}
        structureDict['pdbName'] = names_list[i]
        structureDict['size'] = sizes_list[i]
        structureDict['sequenceList'] = sequence_lists[i]
        structureDict['pdbNamesWithChainsList'] = pdb_names_with_chains_lists[i]
        structures_dicts[full_names_list[i]] = structureDict
    return structures_dicts


def list_creation(file_name):
    """
    :param file_name: PSSM file
    :return: tuple(names_list,sizes_list)
    names_list = list of all the chains's name in the file
    sizes_list = list of all the chains's number of amino acids in the file
    """
    names_list = []
    full_names_list = []
    pdb_names_with_chains_lists = []
    sizes_list = []
    sequence_lists = [[]]
    file1 = open(file_name, 'r')
    last_chain_name = ''
    line = file1.readline().split()
    cnt = 0
    seq = ''
    while len(line) != 0:  # end of file
        cnt += 1
        if len(line) == 1:  # in chain header line
            sequence_lists[len(sequence_lists) - 1].append(seq)
            sizes_list.append(cnt)
            full_name = line[0][1:]
            full_names_list.append(full_name)
            pdb_name, pdb_names_with_chains_list = list_creation_util(full_name)
            names_list.append(pdb_name)
            pdb_names_with_chains_lists.append(pdb_names_with_chains_list)
            try:
                if len(pdb_names_with_chains_lists) > 1:
                    assert (len(sequence_lists[len(sequence_lists) - 1]) == len(
                        pdb_names_with_chains_lists[len(pdb_names_with_chains_lists) - 2]))
                    assert (sizes_list[len(sizes_list) - 1]) == sum(
                        [len(seq) for seq in sequence_lists[len(sequence_lists) - 1]])
            except:
                print(pdb_names_with_chains_list)
                print(sequence_lists[len(sequence_lists) - 1])
                raise Exception(pdb_name)
            sequence_lists.append([])
            cnt = -1
            seq = ''
            last_chain_name = ''
        else:
            if last_chain_name != line[0]:  # switching chains
                last_chain_name = line[0]
                if len(seq) != 0:
                    sequence_lists[len(sequence_lists) - 1].append(seq)
                seq = ''
            seq = seq + line[2]  # not chain's name
        line = file1.readline().split()
    sizes_list.append(cnt)
    sequence_lists[len(sequence_lists) - 1].append(seq)
    sizes_list = sizes_list[1:]  # first one is redundent
    sequence_lists = sequence_lists[1:]
    file1.close()
    return names_list, sizes_list, sequence_lists, full_names_list, pdb_names_with_chains_lists


def make_cath_df(filename, columns_number):
    """
    :param filename: cath-domain-list file
    :param columns_number: the number of columns to consider with the cath classification not include the cath domain name
    :return: dataframe of all the chains in the file and their cath classification divide to 4 different columns
    """

    df = pd.read_csv(filename, skiprows=16, header=None, delimiter=r"\s+")
    df = df.iloc[:, 0:columns_number + 1]
    cath_columns = ["n" + str(i) for i in range(1, columns_number + 1)]
    df.columns = ['chain'] + cath_columns
    df['chain'] = df['chain'].apply(lambda x: x[0:5])
    return df


def make_cath_df_new(cath_path):
    """
    :param filename: cath-domain-list file
    :param columns_number: the number of columns to consider with the cath classification not include the cath domain name
    :return: dataframe of all the chains in the file and their cath classification divide to 4 different columns
    """
    file = open(cath_path, 'r')
    lines = file.readlines()
    structures_names = [line[0:5] for line in lines]
    structures_numbers = [line[5:7] for line in lines]
    c0 = [line.split(" ")[2].split(".")[0] for line in lines]
    c1 = [line.split(" ")[2].split(".")[1] for line in lines]
    c2 = [line.split(" ")[2].split(".")[2] for line in lines]
    c3 = [line.split(" ")[2].split(".")[3] for line in lines]
    data = {
        'chain': structures_names,
        'number': structures_numbers,
        'c0': c0,
        'c1': c1,
        'c2': c2,
        'c3': c3,
    }
    df = pd.DataFrame(data)
    print(df)
    return df


def get_all_cath_classifications_for_chain(cath_df, chain_name, columns_Number):
    df = cath_df[cath_df['chain'] == chain_name]
    my_list = df.values.tolist()
    only_classification_list = [l[2:2 + columns_Number] for l in my_list]
    return only_classification_list


def add_classifications_for_dict(cath_df, structures_dicts, columns_Number):
    for key in structures_dicts.keys():
        structure_dict = structures_dicts[key]
        classifications_lists = []
        for i in range(len(structure_dict['pdbNamesWithChainsList'])):
            if structure_dict['in_or_not_in_cath_list'][i]:
                classifications_lists.append(
                    get_all_cath_classifications_for_chain(cath_df, structure_dict['pdbNamesWithChainsList'][i],
                                                           columns_Number))
            else:
                classifications_lists.append(None)
        structure_dict['classifications_lists'] = classifications_lists


def find_chains_in_cath(cath_df, structures_dicts):
    set_ofchains_names = set(cath_df['chain'])
    for full_pdb_name, structure_dict in structures_dicts.items():
        in_or_not_in_cath_list = [structure_dict['pdbNamesWithChainsList'][j] in set_ofchains_names for j in
                                  range(len(structure_dict['pdbNamesWithChainsList']))]
        structure_dict['in_or_not_in_cath_list'] = in_or_not_in_cath_list


def DivideToStructuresInAndNotInCath(cath_df, structures_dicts):
    set_of_chains_names = set(cath_df['chain'])
    in_cath = []
    not_in_cath = []
    for full_pdb_name, structure_dict in structures_dicts.items():
        sequences_in_cath = []
        pdb_names_with_chains_in = []
        for i in range(len(structure_dict['pdbNamesWithChainsList'])):
            if structure_dict['pdbNamesWithChainsList'][i] in set_of_chains_names:
                sequences_in_cath.append(structure_dict['sequenceList'][i])
                pdb_names_with_chains_in.append(structure_dict['pdbNamesWithChainsList'][i])
        structure_dict['sequences_in_cath'] = sequences_in_cath
        structure_dict['pdb_names_with_chains_in'] = pdb_names_with_chains_in
        if len(pdb_names_with_chains_in) >= 1:
            in_cath.append(full_pdb_name)
        else:
            not_in_cath.append(full_pdb_name)
    return in_cath, not_in_cath


def count_in_cath(cath_df, structures_dicts):
    set_of_chains_names = set(cath_df['chain'])
    cnt = 0
    in_cath = []
    not_in_cath = []
    for full_pdb_name, structure_dict in structures_dicts.items():
        is_in_cath = False
        for i in range(len(structure_dict['pdbNamesWithChainsList'])):
            if structure_dict['pdbNamesWithChainsList'][i] in set_of_chains_names:
                is_in_cath = True
                break
        if is_in_cath:
            cnt += 1
            in_cath.append(full_pdb_name)
        else:
            not_in_cath.append(full_pdb_name)
    return in_cath, not_in_cath, cnt


def neighbor_mat(df, name_list, seqList, columns_number):
    """
    :param df: cath data frame as it return from the func make_cath_df
    :param lst: list of chains
    :param columns_number: the number of columns to consider with the cath classification not include the cath domain name
    :return: matrix. mat[i][j] == 1 if there is connection between chain i and chain j
    """
    # generate the graph using CATH.
    cath_columns = ["n" + str(i) for i in range(1, columns_number + 1)]
    not_in_cath = set()
    n = len(name_list)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            similarity = df[df['chain'].isin([name_list[i], name_list[j]])]
            if len(similarity['chain'].unique()) == 1:
                if (similarity['chain'].unique()[0] == name_list[i]):
                    not_in_cath.add(name_list[j])
                else:
                    not_in_cath.add(name_list[i])
            else:
                similarity = similarity.groupby(by=cath_columns)
                for name, group in similarity:
                    if len(group['chain'].unique()) == 2:
                        mat[i][j] = mat[j][i] = 1
                        break
    # calculate the sequence identity
    for i in range(n):
        for j in range(i + 1, n):
            if (name_list[i] in not_in_cath):
                score = calculate_identity(seqList[i], seqList[j])
                mat[i][j] = mat[j][i] = 1
    return mat


def comapre_classifications(c1, c2):
    for i in range(len(c1)):
        if c1[i] != c2[i]:
            return False
    return True


def is_similiar_chains(structure_dict1, structure_dict2):
    connected = False
    for k in range(len(structure_dict1['pdbNamesWithChainsList'])):
        for l in range(len(structure_dict2['pdbNamesWithChainsList'])):
            if structure_dict1['in_or_not_in_cath_list'][k] and structure_dict2['in_or_not_in_cath_list'][
                l]:  # both chains in cath
                all_classifications1 = structure_dict1['classificationsLists'][k]
                all_classifications2 = structure_dict2['classificationsLists'][l]
                for c1 in all_classifications1:
                    for c2 in all_classifications2:
                        if comapre_classifications(c1, c2):
                            connected = True
            else:  # at least one of the chains not in cath
                if calculate_identity(structure_dict1['sequenceList'][k], structure_dict2['sequenceList'][l]) > 0.5:
                    connected = True
    return connected


def neighbor_mat_new(structuers_dictionaries):
    """
    :param df: cath data frame as it return from the func make_cath_df
    :param lst: list of chains
    :param columns_number: the number of columns to consider with the cath classification not include the cath domain name
    :return: matrix. mat[i][j] == 1 if there is connection between chain i and chain j
    """
    # generate the graph using CATH.
    n = len(structuers_dictionaries)
    print(n)
    structuers_dictionaries_values = [structuers_dictionaries[key] for key in structuers_dictionaries.keys()]
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            structureDict1 = structuers_dictionaries_values[i]
            structureDict2 = structuers_dictionaries_values[j]
            if is_similiar_chains(structureDict1, structureDict2):
                mat[i][j] = mat[j][i] = 1
    return mat


def create_related_chainslist(number_of_components, labels):
    """
    :param number_of_components: number of component = x => 0<=label values<x
    :param labels: labels
    :return: RelatedChainslist: RelatedChainslist[i] = list of all the chain index's which has the label i
    """
    related_chains_lists = [[] for _ in range(number_of_components)]
    for i in range(len(labels)):
        related_chains_lists[labels[i]].append(i)
    return related_chains_lists


def create_cluster_sizes_list(related_chainslists, sizeList):
    """
    :param related_chainslists:  relatedChainslist[i] = list of all the chain index's which has the label i
    :param relatedChainslists:  sizeList- list of all the chains's size
    :return: list of tuples (clusterIndex,size)
    """
    cluster_sizes = []
    for i in range(len(related_chainslists)):
        my_sum = 0
        for index in related_chainslists[i]:
            my_sum += sizeList[index]
        cluster_sizes.append((i, my_sum))
    return cluster_sizes


def divide_clusters(cluster_sizes):
    """
    :param cluster_sizes: list of tuples (clusterIndex,size)
    :return:  sublists,sublists_sum
    divide the list into 5 sublists such that the sum of each cluster sizes in the sublist is as close as possible
    """
    sublists = [[] for i in range(5)]
    sublists_sum = [0 for i in range(5)]
    cluster_sizes.sort(reverse=True, key=lambda x: x[1])  # Sort the clusters by size descending order.
    for tup in cluster_sizes:
        min_cluster_index = sublists_sum.index(min(sublists_sum))  # find the cluster with the minimal sum
        sublists_sum[min_cluster_index] += tup[1]
        sublists[min_cluster_index].append(tup[0])
    return sublists, sublists_sum


def cluster_to_chain_list(cluster_id, related_chains_lists, name_list):
    """
    :param cluster_id: list of chain indexs
    :param related_chains_lists: relatedChainslist[i] = list of all the chain index's in cluster i
    :param  name_list = list of all the chains's name in the file
    :return: chain_list = list of all chain names in the cluster
    """
    cluster = related_chains_lists[cluster_id]  # get the chains in the cluster
    chain_list = [name_list[i] for i in cluster]
    return chain_list


def sublists_to_chain_lists(sublists, related_chains_lists, name_list):
    """
    :param sublists: sublists[i] = all the clusters in sublist i
    :param related_chains_lists: relatedChainslist[i] = list of all the chain index's in cluster i
    :return: chain_lists: ChainLists[i] = list of all the chains in cluster i
    """
    chain_lists = [[] for i in range(len(sublists))]
    for i in range(len(sublists)):
        for clusterId in sublists[i]:
            chain_lists[i] += cluster_to_chain_list(clusterId, related_chains_lists, name_list)
    return chain_lists


def chain_lists_to_chain_index_dict(chain_lists):
    """
    :param chain_lists: chainLists[i] = list of all the chains in cluster i
    :return: chain_dict: chain_dict[chain_name] = index of chain cluster(i if chain in ChainLists[i])
    """
    chain_dict = {}
    for i in range(len(chain_lists)):
        for chain_name in chain_lists[i]:
            chain_dict[chain_name] = i
    return chain_dict


def divide_pssm(chain_dict, full_pssm_file_path):
    """
    :param chain_dict: chainDict[chainName] = index of chain cluster(i if chain in ChainLists[i])
    create len(chainLists) txt files. the i txt file contains the chains in chainLists[i]
    """

    filesList = [open(os.path.join(paths.PSSM_path, f"PSSM{str(i)}.txt"), 'w') for i in range(5)]
    pssm_file = open(full_pssm_file_path, 'r')
    lines = pssm_file.readlines()
    fillIndex = -1  # fillIndex = i -> we now write to PSSMi.txt
    for line in lines:
        if line[0] == '>':  # header line
            fillIndex = chain_dict[line[1:len(line) - 1]]
        filesList[fillIndex].write(line)
    for i in range(5):
        filesList[i].close()
    pssm_file.close()


def calclulate_indexes_of_scc(homologous_labels, scc_number):
    indexes = np.where(homologous_labels == scc_number)
    return indexes[0]


def calculate_ratio_from_indexes(mat_homologous, indexes):
    selected_matrix = mat_homologous[np.ix_(indexes, indexes)]
    avarage_number_of_edges_for_structure = (np.sum(selected_matrix) / len(indexes))
    ratio = avarage_number_of_edges_for_structure / len(indexes)
    return ratio


def calculate_homologous_ratio_for_scc(homologous_labels, mat_homologous, scc_number):
    indexes = calclulate_indexes_of_scc(homologous_labels, scc_number)
    return calculate_ratio_from_indexes(mat_homologous, indexes)


def calculate_ratio_for_fold(homologous_labels, mat_homologous, sublists, fold_num):
    indexes_lists = [calclulate_indexes_of_scc(homologous_labels, sccNumber) for sccNumber in sublists[fold_num]]
    total_indexes = np.concatenate(indexes_lists)
    ratio = calculate_ratio_from_indexes(mat_homologous, total_indexes)
    return ratio
