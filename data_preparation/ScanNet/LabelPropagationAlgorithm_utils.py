import os

import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import pickle
import subprocess
import paths

import numpy as np
import pandas as pd
from models.ScanNet_Ub.preprocessing.sequence_utils import load_FASTA, num2seq
from db_creation_scanNet_utils import save_as_pickle, load_as_pickle
import pdb


def split_receptors_into_individual_chains(pssm_content_file_path, asa_pssm_content_file_path):
    f = open(pssm_content_file_path, 'r')
    fAsa = open(asa_pssm_content_file_path, 'r')
    lines = f.readlines()
    asa_lines = fAsa.readlines()
    chains_keys = []
    chains_sequences = []
    chains_labels = []
    chains_asa_values = []
    chain_names = []
    chain_key = lines[0][1:-1]
    chains_keys.append(chain_key)
    chain_seq = ''
    chain_labels = []
    chain_asa_values = []
    chain_name = None
    for i in range(1, len(lines)):
        line = lines[i]
        asa_line = asa_lines[i]
        if line[0] == '>':
            chains_sequences.append(chain_seq)
            chains_asa_values.append(chain_asa_values)
            chains_labels.append(chain_labels)
            assert len(chain_asa_values) == len(chain_labels)
            chain_seq = ''
            chain_asa_values = []
            chain_labels = []
            chain_key = line[1:-1]
            chains_keys.append(chain_key)
            continue
        elif chains_keys[len(chains_keys) - 1] + '$' + line.split(" ")[0] != chain_name:
            if len(chain_seq) > 0:
                chains_sequences.append(chain_seq)
                chains_labels.append(chain_labels)
                chains_asa_values.append(chain_asa_values)
                chain_seq = ''
                chain_asa_values = []
                chain_labels = []
            chain_name = chains_keys[len(chains_keys) - 1] + '$' + line.split(" ")[0]
            chain_names.append(chain_name)

        asa_info = asa_line.split(" ")
        amino_acid_info = line.split(" ")
        chain_seq += (amino_acid_info[2])
        chain_labels.append(amino_acid_info[3][:-1])
        chain_asa_values.append(float(asa_info[3][:-1]))

    chains_sequences.append(chain_seq)
    chains_labels.append(chain_labels)
    chains_asa_values.append(chain_asa_values)
    assert (len(chain_names) == len(chains_sequences) == len(chains_labels) == len(chains_asa_values))
    f.close()
    return chains_keys, chains_sequences, chains_labels, chain_names, lines, chains_asa_values


def create_cluster_participants_indexes(cluster_indexes):
    clusters_participants_list = []
    for i in range(np.max(cluster_indexes) + 1):
        clusters_participants_list.append(np.where(cluster_indexes == i)[0])
    return clusters_participants_list


def aggregate_cluster_sequences(chains_sequences, clusters_participants_list, index):
    sequences = [chains_sequences[i] for i in clusters_participants_list[index]]
    return sequences


def apply_mafft(sequences, mafft, go_penalty=1.53,
                ge_penalty=0.0, name=None, numeric=False, return_index=True, high_accuracy=True):
    if name is None:
        name = '%.10f' % np.random.rand()
    input_file = os.path.join(paths.tmp_path, 'tmp_%s_unaligned.fasta' % name)
    output_file = os.path.join(paths.tmp_path, 'tmp_%s_aligned.fasta' % name)
    instruction_file = os.path.join(paths.tmp_path, 'tmp_%s.sh' % name)
    with open(input_file, 'w') as f:
        for k, sequence in enumerate(sequences):
            f.write('>%s\n' % k)
            f.write(sequence + '\n')
    if high_accuracy:
        command = '%s  --amino --localpair --maxiterate 1000 --op %s --ep %s %s > %s' % (
            mafft, go_penalty, ge_penalty, input_file, output_file)
    else:
        command = '%s  --amino --auto --op %s --ep %s %s > %s' % (
            mafft, go_penalty, ge_penalty, input_file, output_file)
    print(command)
    with open(instruction_file, 'w') as f:
        f.write(command)
    os.system('sh %s' % instruction_file)

    alignment = load_FASTA(
        output_file, drop_duplicates=False)[0]
    if return_index:
        is_gap = alignment == 20
        index = np.cumsum(1 - is_gap, axis=1) - 1
        index[is_gap] = -1

    if not numeric:
        alignment = num2seq(alignment)
    os.system('rm %s' % input_file)
    os.system('rm %s' % output_file)
    os.system('rm %s' % instruction_file)

    if return_index:
        return alignment, index
    else:
        return alignment


def apply_mafft_for_all_clusters(chains_sequences, clusters_participants_list, path_to_mafft_exec):
    clusters_dict = dict()
    aligments = []
    indexes = []
    for i in range(len(clusters_participants_list)):
        sequences = aggregate_cluster_sequences(chains_sequences, clusters_participants_list, i)
        aligment, index = apply_mafft(sequences, path_to_mafft_exec)
        aligments.append(aligment)
        indexes.append(index)
    clusters_dict['aligments'] = aligments
    clusters_dict['indexes'] = indexes
    return clusters_dict


def create_propagated_labels_for_cluster(index, chains_labels, cluster_participants_list, chains_asa_values):
    number_of_participants = index.shape[0]
    msa_length = index.shape[1]
    assert (number_of_participants == len(cluster_participants_list))
    new_labels = [[] for _ in range(number_of_participants)]
    labels_after_aligment = [[0 for _ in range(msa_length)] for _ in range(number_of_participants)]
    for i in range(number_of_participants):
        current_labels = chains_labels[cluster_participants_list[i]]
        indexs_of_parcipitant = index[i]
        for j in range(msa_length):
            if indexs_of_parcipitant[j] != -1:  # not a gap
                labels_after_aligment[i][j] = int(current_labels[indexs_of_parcipitant[j]])

    consensus = [max([labels_after_aligment[i][j] for i in range(number_of_participants)]) for j in range(msa_length)]
    for i in range(number_of_participants):
        chain_index = cluster_participants_list[i]
        indexs_of_parcipitant = index[i]
        threshold = min(0.2, 0.75 * max(chains_asa_values[chain_index]))
        # print("i = ", i)
        for j in range(msa_length):
            # print("j = ", j)
            if indexs_of_parcipitant[j] != -1:  # not a gap
                if chains_asa_values[chain_index][len(new_labels[i])] > threshold:
                    new_labels[i].append(consensus[j])
                else:
                    new_labels[i].append(chains_labels[chain_index][len(new_labels[i])])
    return new_labels


def find_chain_names_for_cluster(clusters_participants_list, chain_names, i):
    cluster_chains_names = [chain_names[j] for j in clusters_participants_list[i]]
    return cluster_chains_names


def find_chain_names_for_clusters(clusters_participants_list, chain_names):
    print(chain_names)
    clusters_chains_names = [find_chain_names_for_cluster(clusters_participants_list, chain_names, i) for i in
                             range(len(clusters_participants_list))]
    return clusters_chains_names


def create_propagated_pssm_file(clusters_dict, chains_labels, clusters_participants_list,
                                chains_sequences, chain_names, lines, chains_asa_values, propagated_file_path):
    num_of_clusters = len(clusters_dict['indexes'])
    num_of_chains = len(chains_sequences)
    new_labels = [None for i in range(num_of_chains)]
    clusters_chains_names = find_chain_names_for_clusters(clusters_participants_list, chain_names)
    # clustersIndexes = [findIndexesForCluster(clusterChainNames) for clusterChainNames in clusters_chains_names]
    for i in range(num_of_clusters):
        cluster_new_labels = create_propagated_labels_for_cluster(clusters_dict['indexes'][i], chains_labels,
                                                                  clusters_participants_list[i], chains_asa_values)
        for j in range(len(clusters_participants_list[i])):
            new_labels[clusters_participants_list[i][j]] = cluster_new_labels[j]

    propagated_file = open(propagated_file_path, 'w')
    chain_index = -1
    chain_name = None
    for line in lines:
        if line[0] == '>':
            chains_key = line[1:-1]
        else:
            if chains_key + '$' + line.split(" ")[0] != chain_name:
                chain_name = chains_key + '$' + line.split(" ")[0]
                chain_index += 1
                amino_acid_num = 0
            splited_line = line.split(" ")
            splited_line[-1] = str(new_labels[chain_index][amino_acid_num]) + '\n'
            line = " ".join(splited_line)
            amino_acid_num += 1
        propagated_file.write(line)
    propagated_file.close()


def create_quantile_asa_dicts(lines):
    amino_acid_asa_dict = dict()
    for line in lines:
        if line[0] != '>':
            splitted_line = line.split(" ")
            asa_val = splitted_line[3][:-1]
            amino_acid_char = splitted_line[2]
            if amino_acid_char not in amino_acid_asa_dict:
                amino_acid_asa_dict[amino_acid_char] = []
            amino_acid_asa_dict[amino_acid_char].append(float(asa_val))
    quentile_asa_amino_acid_dict = dict()

    for amino_acid_char in amino_acid_asa_dict.keys():
        quantile5 = np.percentile(amino_acid_asa_dict[amino_acid_char], 5)
        quantile95 = np.percentile(amino_acid_asa_dict[amino_acid_char], 95)
        quentile_asa_amino_acid_dict[amino_acid_char] = (quantile5, quantile95)
    return quentile_asa_amino_acid_dict


def normalize_value(current_val, quantile5, quantile95):
    if current_val <= quantile5:
        return 0
    if current_val >= quantile95:
        return 1
    normalize_value = (current_val - quantile5) / (quantile95 - quantile5)
    return normalize_value


def normalize_asa_data(full_asa_pssm_path, normalized_asa_path):
    f = open(full_asa_pssm_path, 'r')
    lines = f.readlines()
    f.close()
    quentile_asa_amino_acid_dict = create_quantile_asa_dicts(lines)
    normalize_asa_pssm_content_file = open(normalized_asa_path, 'w')
    for line in lines:
        if line[0] == '>':
            normalize_asa_pssm_content_file.write(line)
        else:
            splitted_line = line.split(" ")
            asa_val = float(splitted_line[3][:-1])
            amino_acid_char = splitted_line[2]
            normalized_asa_value = normalize_value(asa_val, quentile_asa_amino_acid_dict[amino_acid_char][0],
                                                   quentile_asa_amino_acid_dict[amino_acid_char][1])
            splitted_line[3] = str(normalized_asa_value) + '\n'
            new_line = " ".join(splitted_line)
            normalize_asa_pssm_content_file.write(new_line)
    normalize_asa_pssm_content_file.close()
