import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import paths
from data_preparation.ScanNet.LabelPropagationAlgorithm_utils import split_receptors_into_individual_chains
# from UBDModel import LabelPropagationAlgorithm
# from UBDModel import Uniprot_utils


def from_binding_residues_string_to_hot_one_encoding(binding_residues_string):
    hot_one_encoding = np.zeros(75)
    if binding_residues_string == '\n':
        return hot_one_encoding
    binding_residues = binding_residues_string[:-1].split('+')
    binding_residues_numbers = ["".join([char for char in bindingResidue if char.isdigit()]) for bindingResidue in
                              binding_residues]
    binding_residues_numbers_as_ints = [int(binding_residues_number) for binding_residues_number in binding_residues_numbers]
    for num in binding_residues_numbers_as_ints:
        hot_one_encoding[num - 1] = 1
    return hot_one_encoding


def create_hot_one_encodings(ubiquitin_binding_residues):
    ubiquitin_binding_encodings = [None for i in range(len(ubiquitin_binding_residues))]
    for i in range(len(ubiquitin_binding_encodings)):
        ubiquitin_binding_encodings[i] = from_binding_residues_string_to_hot_one_encoding(ubiquitin_binding_residues[i])
    return ubiquitin_binding_encodings


def apply_PCA(hot_one_encodings, numer_of_components):
    pca = PCA(n_components=numer_of_components)
    pca.fit(hot_one_encodings)
    transformed_data = pca.transform(hot_one_encodings)
    return transformed_data


def Kmeans(transformed_data):
    n_clusters = 5  # Number of clusters
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(transformed_data)
    labels = kmeans.labels_
    return kmeans, labels


def gaussian_mixture(transformed_data):
    n_components = 5  # Number of clusters
    gmm = GaussianMixture(n_components=n_components)
    gmm.fit(transformed_data)
    labels = gmm.fit_predict(transformed_data)
    return gmm, labels


def plotResults(transformed_data, labels, gmm):
    cluster_centers = gmm.means_
    plt.scatter(transformed_data[:, 0], transformed_data[:, 1], c=labels, cmap='viridis')
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='X', s=200, color='red', label='Cluster Centers')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA + gmm Clustering')
    plt.show()


# plotResults(transformed_data, labels, gmm)

def unpickle(path):
    with open(path, 'rb') as file:
        # Load the pickled object
        myObject = pickle.load(file)
        return myObject


def find_index_in_tested(receptor_name, unpropagated_predictions):
    try:
        index = unpropagated_predictions['list_origins'].index(receptor_name)
    except:
        return None
    return index


def all_valid_indexes(mono_ubiquitin_receptors_names, unpropagated_predictions):
    valid_list = [i for i in range(len(mono_ubiquitin_receptors_names)) if
                 find_index_in_tested(mono_ubiquitin_receptors_names[i], unpropagated_predictions) is not None]
    return valid_list


def calculate_positives_avarage_predicted_probability_for_receptor(receptor_name, unpropagated_predictions):
    index = find_index_in_tested(receptor_name, unpropagated_predictions)
    predictions_index = find_index_in_tested(receptor_name, unpropagated_predictions)
    if index is None or predictions_index is None:
        return None
    labels_list = unpropagated_predictions['list_labels'][index]
    predictions_list = unpropagated_predictions['list_predictions'][predictions_index]

    positives_predictions = [predictions_list[i] for i in range(len(labels_list)) if
                            labels_list[i] == 2 or labels_list[i] == 3]
    average_positives_predictions = sum(positives_predictions) / len(positives_predictions)
    return average_positives_predictions


def calculate_positives_average_predicted_probability(mono_ubiquitin_receptors_names, unpropagated_predictions):
    average_positives_predictions_dict = dict()
    for i in range(len(mono_ubiquitin_receptors_names)):
        average_positives_predictions = calculate_positives_avarage_predicted_probability_for_receptor(
            mono_ubiquitin_receptors_names[i], unpropagated_predictions)
        if average_positives_predictions is not None:
            average_positives_predictions_dict[mono_ubiquitin_receptors_names[i]] = average_positives_predictions
    return average_positives_predictions_dict


def calculate_is_covalent_bond_for_receptor(receptor_name, unpropagated_predictions, mono_ubiquitin_splitted_lines,
                                       mono_ubiq_index):
    index = find_index_in_tested(receptor_name, unpropagated_predictions)
    predictions_index = find_index_in_tested(receptor_name, unpropagated_predictions)
    if index is None or predictions_index is None:
        return None
    # remove endLine from last residue
    mono_ubiquitin_splitted_lines[mono_ubiq_index][3] = mono_ubiquitin_splitted_lines[mono_ubiq_index][3][:-1]
    ubiquitins_bounded_residues_strings = mono_ubiquitin_splitted_lines[mono_ubiq_index][3].split('//')
    C_terminus_amino_acid = 'G75'
    for ubiquitin_bounded_residues_string in ubiquitins_bounded_residues_strings:
        ubiquitin_bounded_residues = ubiquitin_bounded_residues_string.split('+')
        if C_terminus_amino_acid in ubiquitin_bounded_residues:
            return True
    return False


def calculate_is_covalent_bond_for_receptors(mono_ubiquitin_receptors_names):
    is_covalent_bond_dict = dict()
    for i in range(len(mono_ubiquitin_receptors_names)):
        is_covalent_bond = calculate_is_covalent_bond_for_receptor(mono_ubiquitin_receptors_names[i], unpropagated_predictions,
                                                            mono_ubiquitin_splitted_lines,
                                                            i)
        if is_covalent_bond is not None:
            is_covalent_bond_dict[mono_ubiquitin_receptors_names[i]] = is_covalent_bond
    return is_covalent_bond_dict


def calculate_clusters_for_receptors(mono_ubiquitin_receptors_names, ubiquitin_binding_residues, unpropagated_predictions):
    receptors_clusters_dict = dict()
    hot_one_encodings = create_hot_one_encodings(ubiquitin_binding_residues)
    transformed_data = apply_PCA(hot_one_encodings, 10)
    gmm, labels = gaussian_mixture(transformed_data)
    list_of_valid_indexes = all_valid_indexes(mono_ubiquitin_receptors_names, unpropagated_predictions)
    for index in list_of_valid_indexes:
        receptors_clusters_dict[mono_ubiquitin_receptors_names[index]] = labels[index]
    return receptors_clusters_dict


def make_chain_dict(chain_names):
    chainDict = dict()
    for chainName in chain_names:
        receptorName = chainName.split('$')[0]
        chainId = chainName.split('$')[1]
        if receptorName in chainDict:
            chainDict[receptorName].append(chainId)
        else:
            chainDict[receptorName] = []
            chainDict[receptorName].append(chainId)
    return chainDict


def look_for_class_in_string_util(dicription_string, class_dict_for_receptor):
    lookup_strings_dict = {'e1': ['e1', 'activating'], 'e2': ['e2', 'conjugating'], 'e3|e4': ['e3', 'e4', 'ligase'],
                         'deubiquitylase': ['deubiquitylase', 'hydrolase', 'deubiquitinating', 'deubiquitinase',
                                            'protease', 'deubiquitin', 'isopeptidase', 'peptidase']}
    dicription_string_lower = dicription_string.lower()
    for key in class_dict_for_receptor.keys():
        for lookup_string in lookup_strings_dict[key]:
            if dicription_string_lower.find(lookup_string) != -1:
                class_dict_for_receptor[key] = True


def find_class_for_receptor(pdb_name, chains_names, not_found_tuples_list):
    class_dict_for_receptor = {'e1': False, 'e2': False, 'e3|e4': False, 'deubiquitylase': False}
    for chain_name in chains_names:
        pdb_name_4_letters = pdb_name[:4]
        print((pdb_name_4_letters, chain_name))
        try:
            _, name, _, _, _ = Uniprot_utils.get_chain_organism(pdb_name_4_letters, chain_name)
            look_for_class_in_string_util(name, class_dict_for_receptor)
        except:
            print('Exception! ')
            not_found_tuples_list.append((pdb_name_4_letters, chain_name))
    return class_dict_for_receptor


def find_class_for_receptors(pssm_content_path,asa_content_path,mono_ubiquitin_receptors_names):
    _, _, _, chain_names, _, _ = split_receptors_into_individual_chains(
        pssm_content_path, asa_content_path)
    chain_dict = make_chain_dict(chain_names)
    not_found_tuples_list = []
    e1Dict = dict()
    e2Dict = dict()
    e3e4Dict = dict()
    deubiquitylaseDict = dict()
    for i in range(len(mono_ubiquitin_receptors_names)):
        chainIdsForReceptor = chain_dict[mono_ubiquitin_receptors_names[i]]
        class_dict_for_receptor = find_class_for_receptor(mono_ubiquitin_receptors_names[i], chainIdsForReceptor,
                                                    not_found_tuples_list)
        e1Dict[mono_ubiquitin_receptors_names[i]] = class_dict_for_receptor['e1']
        e2Dict[mono_ubiquitin_receptors_names[i]] = class_dict_for_receptor['e2']
        e3e4Dict[mono_ubiquitin_receptors_names[i]] = class_dict_for_receptor['e3|e4']
        deubiquitylaseDict[mono_ubiquitin_receptors_names[i]] = class_dict_for_receptor['deubiquitylase']
    return e1Dict, e2Dict, e3e4Dict, deubiquitylaseDict, not_found_tuples_list



if __name__ == 'main':
    summaryFile = open(os.path.join(paths.ImerFiles_path, 'Integrated_summaryLog.txt'), 'r')
    lines = summaryFile.readlines()
    summaryFile.close()
    splittedLines = [line.split('$') for line in lines]
    mono_ubiquitin_splitted_lines = [splittedLine for splittedLine in splittedLines if splittedLine[2] == '1']
    mono_ubiquitin_receptors_names = [splittedLine[0] for splittedLine in mono_ubiquitin_splitted_lines]
    # ubiquitinBindingResidues = [splittedLine[3] for splittedLine in mono_ubiquitin_splitted_lines]
    # unpropagatedPath = 'C:\\Users\\omriy\\UBDAndScanNet\\UBDModel\\model_predictions\\predictions_ubiquitin_ScanNet_PUI_retrained_0108.pkl'
    # unpropagated_predictions = unpickle(unpropagatedPath)
    # receptorsClustersDict = calculate_clusters_for_receptors(mono_ubiquitin_receptors_names, ubiquitinBindingResidues,
    #                                                       unpropagated_predictions)
    # isCovalentBondDict = calculate_is_covalent_bond_for_receptors(mono_ubiquitin_receptors_names)
    pssm_content_path = os.path.join(paths.PSSM_path, 'propagatedPssmWithAsaFile.txt')
    asa_content_path = os.path.join(paths.ASA_path, 'normalizedFullASAPssmContent.txt')
    e1Dict, e2Dict, e3e4Dict, deubiquitylaseDict, notFoundTuplesList = find_class_for_receptors(pssm_content_path,
                                                                                                asa_content_path,
                                                                                                mono_ubiquitin_receptors_names)
    print(f'e1Dict: {e1Dict}')
    print(f'e2Dict: {e2Dict}')
    print(f'e3e4Dict: {e3e4Dict}')
    print(f'deubiquitylaseDict: {deubiquitylaseDict}')
    print(f'notFoundTuplesList: {notFoundTuplesList}')

    # averagePositivesPredictionsDict = calculate_positives_average_predicted_probability(mono_ubiquitin_receptors_names,
    #                                                                                 unpropagated_predictions)
    # classificationList = [e1Dict, e2Dict, e3e4Dict, deubiquitylaseDict, notFoundTuplesList]
    # LabelPropagationAlgorithm.saveAsPickle(classificationList, 'classificationList')
    # data = {'ClusterNumber': receptorsClustersDict, 'isCovalent': isCovalentBondDict,
    #         'averagePositivesPredictions': averagePositivesPredictionsDict, 'e1': e1Dict, 'e2': e2Dict, 'e3|e4': e3e4Dict,
    #         'deubiquitylase': deubiquitylaseDict}
    # df = pd.DataFrame(data)
    # df.index.name = 'ReceptorName'
    # excelFileName = 'orientationAnalysis.xlsx'
    # df.to_excel(excelFileName, index=True)
    # print('Done')
