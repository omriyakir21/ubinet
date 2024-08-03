import os
import pickle
import subprocess
import sys
import paths

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(paths.current_dir, '..'))

import numpy as np
import pandas as pd

ubuntu = False

if ubuntu:
    rootPath = '/mnt/c/Users/omriy/ubinet/'
else:
    rootPath = paths.current_dir


from ScanNet_Ub.preprocessing.sequence_utils import load_FASTA, num2seq
from db_creation_scanNet_utils import save_as_pickle,load_as_pickle



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
    return np.array(chains_keys), np.array(chains_sequences), np.array(chains_labels), chain_names, lines, chains_asa_values


def cluster_sequences(list_sequences, seqid=0.95, coverage=0.8, covmode='0'):
    path2mmseqs = '/home/omriyakir21/MMseqs2/build/bin//mmseqs'
    path2mmseqstmp = '/mnt/c/Users/omriy/UBDAndScanNet/UBDModel/mmseqs2'

    rng = np.random.randint(0, high=int(1e6))
    tmp_input = os.path.join(path2mmseqstmp, 'tmp_input_file_%s.fasta' % rng)
    tmp_output = os.path.join(path2mmseqstmp, 'tmp_output_file_%s' % rng)

    with open(tmp_input, 'w') as f:
        for k, sequence in enumerate(list_sequences):
            f.write('>%s\n' % k)
            f.write('%s\n' % sequence)

    command = ('{mmseqs} easy-cluster {fasta} {result} {tmp} --min-seq-id %s -c %s --cov-mode %s' % (
        seqid, coverage, covmode)).format(mmseqs=path2mmseqs, fasta=tmp_input, result=tmp_output, tmp=path2mmseqstmp)
    subprocess.run(command.split(' '))

    with open(tmp_output + '_rep_seq.fasta', 'r') as f:
        representative_indices = [int(x[1:-1]) for x in f.readlines()[::2]]
    cluster_indices = np.zeros(len(list_sequences), dtype=int)
    table = pd.read_csv(tmp_output + '_cluster.tsv', sep='\t', header=None).to_numpy(dtype=int)
    for i, j in table:
        if i in representative_indices:
            cluster_indices[j] = representative_indices.index(i)
    for file in [tmp_output + '_rep_seq.fasta', tmp_output + '_all_seqs.fasta', tmp_output + '_cluster.tsv']:
        os.remove(file)
    save_as_pickle(cluster_indices, path2mmseqstmp + '/clusterIndices.pkl')
    return np.array(cluster_indices), np.array(representative_indices)


if ubuntu:
    chainsKeys, chainsSequences, chainsLabels, chainNames, lines, chainsAsaValues = split_receptors_into_individual_chains(
        rootPath + '/UBDModel/FullPssmContent.txt', rootPath + 'normalizedFullASAPssmContent')
    cluster_indices, representative_indices = cluster_sequences(chainsSequences)
    clusterIndexes = loadPickle(rootPath + 'UBDModel/mmseqs2/clusterIndices.pkl')
else:
    chainsKeys, chainsSequences, chainsLabels, chainNames, lines, chainsAsaValues = split_receptors_into_individual_chains(
        rootPath + '\\UBDModel\\FullPssmContent.txt', rootPath + '\\UBDModel\\normalizedFullASAPssmContent')
    clusterIndexes = loadPickle(rootPath + 'UBDModel\\mmseqs2\\clusterIndices.pkl')

path2mafft = '/usr/bin/mafft'


def createClusterParticipantsIndexes(clusterIndexes):
    clustersParticipantsList = []
    for i in range(np.max(clusterIndexes) + 1):
        clustersParticipantsList.append(np.where(clusterIndexes == i)[0])
    return clustersParticipantsList


clustersParticipantsList = createClusterParticipantsIndexes(clusterIndexes)


def aggragateClusterSequences(chainsSequences, clustersParticipantsList, index):
    sequences = chainsSequences[clustersParticipantsList[index]]
    return sequences


def apply_mafft(sequences, mafft=path2mafft, go_penalty=1.53,
                ge_penalty=0.0, name=None, numeric=False, return_index=True, high_accuracy=True):
    if name is None:
        name = '%.10f' % np.random.rand()
    input_file = 'tmp_%s_unaligned.fasta' % name
    output_file = 'tmp_%s_aligned.fasta' % name
    instruction_file = 'tmp_%s.sh' % name
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


def applyMafftForAllClusters(chainsSequences, clustersParticipantsList):
    clustersDict = dict()
    aligments = []
    indexes = []
    for i in range(len(clustersParticipantsList)):
        sequences = aggragateClusterSequences(chainsSequences, clustersParticipantsList, i)
        aligment, index = apply_mafft(sequences)
        aligments.append(aligment)
        indexes.append(index)
    clustersDict['aligments'] = aligments
    clustersDict['indexes'] = indexes
    return clustersDict


if ubuntu:
    clustersDict = applyMafftForAllClusters(chainsSequences, clustersParticipantsList)
    saveAsPickle(clustersDict, '/mnt/c/Users/omriy/UBDAndScanNet/UBDModel/mafft/clustersDict')
    clustersDict = loadPickle(rootPath + 'UBDModel/mafft/clustersDict.pkl')
else:
    clustersDict = loadPickle(rootPath + 'UBDModel\\mafft\\clustersDict.pkl')


def createPropagatedLabelsForCluster(index, chainsLabels, clusterParticipantsList, chainsAsaValues):
    numberOfParticipants = index.shape[0]
    msaLength = index.shape[1]
    assert (numberOfParticipants == len(clusterParticipantsList))
    newLabels = [[] for _ in range(numberOfParticipants)]
    labelsAfterAligment = [[0 for _ in range(msaLength)] for _ in range(numberOfParticipants)]
    for i in range(numberOfParticipants):
        currentLabels = chainsLabels[clusterParticipantsList[i]]
        indexsOfParcipitant = index[i]
        for j in range(msaLength):
            if indexsOfParcipitant[j] != -1:  # not a gap
                labelsAfterAligment[i][j] = int(currentLabels[indexsOfParcipitant[j]])

    consensus = [max([labelsAfterAligment[i][j] for i in range(numberOfParticipants)]) for j in range(msaLength)]
    for i in range(numberOfParticipants):
        chainIndex = clusterParticipantsList[i]
        indexsOfParcipitant = index[i]
        threshold = min(0.2, 0.75 * max(chainsAsaValues[chainIndex]))
        # print("i = ", i)
        for j in range(msaLength):
            # print("j = ", j)
            if indexsOfParcipitant[j] != -1:  # not a gap
                if chainsAsaValues[chainIndex][len(newLabels[i])] > threshold:
                    newLabels[i].append(consensus[j])
                else:
                    newLabels[i].append(chainsLabels[chainIndex][len(newLabels[i])])
    return newLabels


# createPropagatedLabelsForCluster(clustersDict['indexes'][1], chainsLabels, clustersParticipantsList[1])

def findChainNamesForCluster(clustersParticipantsList, chainNames, i):
    clusterChainsNames = [chainNames[j] for j in clustersParticipantsList[i]]
    return clusterChainsNames


def findChainNamesForClusters(clustersParticipantsList, chainNames):
    print(chainNames)
    clustersChainsNames = [findChainNamesForCluster(clustersParticipantsList, chainNames, i) for i in
                           range(len(clustersParticipantsList))]
    return clustersChainsNames


# def findIndexesForCluster(clusterChainNames, chainNames):
#     clusterIndexes = [chainNames.index(name) for name in clusterChainNames]
#     return clusterIndexes


def createPropagatedPssmFile(clustersDict, chainsLabels, clustersParticipantsList,
                             chainsSequences, chainNames, lines, chainsAsaValues):
    numOfClusters = len(clustersDict['indexes'])
    numOfChains = len(chainsSequences)
    newLabels = [None for i in range(numOfChains)]
    clustersChainsNames = findChainNamesForClusters(clustersParticipantsList, chainNames)
    # clustersIndexes = [findIndexesForCluster(clusterChainNames) for clusterChainNames in clustersChainsNames]
    for i in range(numOfClusters):
        clusterNewLabels = createPropagatedLabelsForCluster(clustersDict['indexes'][i], chainsLabels,
                                                            clustersParticipantsList[i], chainsAsaValues)
        for j in range(len(clustersParticipantsList[i])):
            newLabels[clustersParticipantsList[i][j]] = clusterNewLabels[j]

    propagatedFile = open('propagatedPssmWithAsaFile0.2', 'w')
    chainIndex = -1
    chainName = None
    for line in lines:
        if line[0] == '>':
            chainsKey = line[1:-1]
        else:
            if chainsKey + '$' + line.split(" ")[0] != chainName:
                chainName = chainsKey + '$' + line.split(" ")[0]
                chainIndex += 1
                aminoAcidNum = 0
            splitedLine = line.split(" ")
            splitedLine[-1] = str(newLabels[chainIndex][aminoAcidNum]) + '\n'
            line = " ".join(splitedLine)
            aminoAcidNum += 1
        propagatedFile.write(line)
    propagatedFile.close()


def createQuantileAsaDicts(lines):
    aminoAcidAsaDict = dict()
    for line in lines:
        if line[0] != '>':
            splittedLine = line.split(" ")
            asaVal = splittedLine[3][:-1]
            aminoAcidChar = splittedLine[2]
            if aminoAcidChar not in aminoAcidAsaDict:
                aminoAcidAsaDict[aminoAcidChar] = []
            aminoAcidAsaDict[aminoAcidChar].append(float(asaVal))
    quentileAsaAminoAcidDict = dict()

    for aminoAcidChar in aminoAcidAsaDict.keys():
        quantile5 = np.percentile(aminoAcidAsaDict[aminoAcidChar], 5)
        quantile95 = np.percentile(aminoAcidAsaDict[aminoAcidChar], 95)
        quentileAsaAminoAcidDict[aminoAcidChar] = (quantile5, quantile95)
    return quentileAsaAminoAcidDict


def normalizeValue(currentVal, quantile5, quantile95):
    if currentVal <= quantile5:
        return 0
    if currentVal >= quantile95:
        return 1
    normalizeValue = (currentVal - quantile5) / (quantile95 - quantile5)
    return normalizeValue


def normalizeASAData(fullAsaPssmContent):
    f = open(fullAsaPssmContent, 'r')
    lines = f.readlines()
    f.close()
    quentileAsaAminoAcidDict = createQuantileAsaDicts(lines)
    normalizeASAPssmContentFile = open('normalizedFullASAPssmContent', 'w')
    for line in lines:
        if line[0] == '>':
            normalizeASAPssmContentFile.write(line)
        else:
            splittedLine = line.split(" ")
            asaVal = float(splittedLine[3][:-1])
            aminoAcidChar = splittedLine[2]
            normalizedAsaValue = normalizeValue(asaVal, quentileAsaAminoAcidDict[aminoAcidChar][0],
                                                quentileAsaAminoAcidDict[aminoAcidChar][1])
            splittedLine[3] = str(normalizedAsaValue) + '\n'
            newLine = " ".join(splittedLine)
            normalizeASAPssmContentFile.write(newLine)
    normalizeASAPssmContentFile.close()


# normalizeASAData('FullAsaPssmContent')

# createPropagatedPssmFile(clustersDict, chainsLabels, clustersParticipantsList, chainsSequences, chainNames, lines,
#                          chainsAsaValues)
