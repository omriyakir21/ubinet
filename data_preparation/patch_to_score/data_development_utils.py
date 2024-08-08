import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import csv
import pickle
from itertools import chain
from plistlib import load
from data_preparation.ScanNet.db_creation_scanNet_utils import save_as_pickle, load_as_pickle, \
    THREE_LETTERS_TO_SINGLE_AA_DICT, aa_out_of_chain
import networkx as nx
import numpy as np
import os
import networkx
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from matplotlib import pyplot as plt
import pandas as pd
from Bio.PDB import MMCIFParser
import seaborn as sns
from sklearn.metrics import auc
from data_preparation.ScanNet.db_creation_scanNet_utils import save_as_pickle, load_as_pickle
import paths

NEGATIVE_SOURCES = set(
    ['Yeast proteome', 'Human proteome', 'Ecoli proteome', 'Celegans proteome', 'Arabidopsis proteome'])

POSITIVE_SOURCES = set(['E1', 'E2', 'E3', 'ubiquitinBinding', 'DUB'])

parser = MMCIFParser()
all_predictions = load_as_pickle(os.path.join(paths.ScanNet_results_path, 'all_predictions_0304_MSA_True.pkl'))
all_predictions_ubiq = all_predictions['dict_predictions_ubiquitin']
all_predictions_ubiq_flatten = [value for values_list in all_predictions_ubiq.values() for value in values_list]
percentile_90 = np.percentile(all_predictions_ubiq_flatten, 90)
server_PDBs = True
DISTANCE_THRESHOLD = 10
indexes = list(range(0, len(all_predictions['dict_resids']) + 1, 1500)) + [len(all_predictions['dict_resids'])]
#to be removed
trainingDataDir = None
ubdPath = None

class SizeDifferentiationException(Exception):
    def __init__(self, uniprotName):
        super().__init__("uniprotName: ", uniprotName, "\n")


class Protein:
    def __init__(self, uniprot_name, plddt_threshold):
        self.uniprot_name = uniprot_name
        self.ubiq_predictions = all_predictions['dict_predictions_ubiquitin'][uniprot_name]
        self.non_ubiq_predictions = all_predictions['dict_predictions_interface'][uniprot_name]
        self.residues = all_predictions['dict_resids'][uniprot_name]
        self.source = self.get_source(all_predictions['dict_sources'][uniprot_name])
        self.plddt_values = self.get_plddt_values()
        self.size = None
        self.graph = nx.Graph()
        self.create_graph(plddt_threshold)
        self.connected_components_tuples = self.creat_connected_components_tuples()

    def get_source(self, source):
        if server_PDBs:
            return source
        if source == 'Human proteome':
            return 'proteome'
        else:
            return source

    def get_structure(self):
        if server_PDBs:
            structurePath = all_predictions['dict_pdb_files'][self.uniprot_name]
        else:
            GoPath = path.GoPath
            typePath = os.path.join(GoPath, self.source)
            if self.source == 'proteome':
                structurePath = os.path.join(typePath, 'AF-' + self.uniprot_name + '-F1-model_v4.cif')
            else:
                structurePath = os.path.join(typePath, self.uniprot_name + '.cif')
        structure = parser.get_structure(self.uniprot_name, structurePath)
        return structure

    def get_plddt_values(self):
        structure = self.get_structure()
        model = structure.child_list[0]
        assert (len(model) == 1)
        for chain in model:
            residues = aa_out_of_chain(chain)
            return np.array([residues[i].child_list[0].bfactor for i in range(len(residues))])

    def create_nodes_for_graph(self, residues, plddt_threshold):
        nodes = []
        if len(residues) != len(self.ubiq_predictions):  # need to skip this protein
            raise SizeDifferentiationException(self.uniprot_name)
        for i in range(len(residues)):
            plddtVal = residues[i].child_list[0].bfactor
            if plddtVal > plddt_threshold and self.ubiq_predictions[i] > percentile_90:
                nodes.append(i)
        return nodes

    def create_edges_for_graph(self, residues, nodes):
        edges = []
        c_alpha_atoms = [residue["CA"] for residue in residues]
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                if c_alpha_distance(c_alpha_atoms[i], c_alpha_atoms[j]) < DISTANCE_THRESHOLD:
                    edges.append((nodes[i], nodes[j]))
        return edges

    def create_graph(self, plddt_threshold):
        structure = self.get_structure()
        model = structure.child_list[0]
        assert (len(model) == 1)
        for chain in model:
            residues = aa_out_of_chain(chain)
            self.size = len(residues)
            nodes = self.create_nodes_for_graph(residues, plddt_threshold)
            valid_residues = [residues[i] for i in nodes]
            edges = self.create_edges_for_graph(valid_residues, nodes)
            self.graph.add_nodes_from(nodes)
            self.graph.add_edges_from(edges)

    def creat_connected_components_tuples(self):
        tuples = []
        connected_components = list(nx.connected_components(self.graph))
        for component_set in connected_components:
            average_ubiq, average_non_ubiq, average_plddt = self.calculate_average_predictions_for_component(
                component_set)
            length = len(component_set)
            tuples.append((length, average_ubiq, average_non_ubiq, average_plddt, list(component_set)))
        return tuples

    def calculate_average_predictions_for_component(self, indexSet):
        indexes = list(indexSet)
        ubiq_predictions = [self.ubiq_predictions[index] for index in indexes]
        non_ubiq_predictions = [self.non_ubiq_predictions[index] for index in indexes]
        plddt_values = [self.plddt_values[index] for index in indexes]
        assert (len(ubiq_predictions) == len(non_ubiq_predictions) == len(plddt_values))
        average_ubiq = sum(ubiq_predictions) / len(ubiq_predictions)
        average_non_ubiq = sum(non_ubiq_predictions) / len(non_ubiq_predictions)
        average_plddt = sum(plddt_values) / len(plddt_values)
        return average_ubiq, average_non_ubiq, average_plddt


def c_alpha_distance(atom1, atom2):
    vector1 = atom1.get_coord()
    vector2 = atom2.get_coord()
    distance = np.sqrt(((vector2[np.newaxis] - vector1[np.newaxis]) ** 2).sum(-1))
    return distance


# # allPredictionsUbiq = {key: allPredictionsUbiq[key] for key in common_keys}
# # dict_resids = {key: allPredictions['dict_resids'][key] for key in common_keys}
# # dict_sequences = {key: allPredictions['dict_sequences'][key] for key in common_keys}
# # dict_sources = {key: allPredictions['dict_sources'][key] for key in common_keys}
# # allPredictions['dict_resids'] = dict_resids
# # allPredictions['dict_sequences'] = dict_sequences
# # allPredictions['dict_sources'] = dict_sources
# # allPredictions['dict_predictions_ubiquitin'] = allPredictionsUbiq
# # saveAsPickle(allPredictions,r'C:\Users\omriy\UBDAndScanNet\newUBD\UBDModel\Predictions\all_predictions_22_3')


def create_patches_dict(i, dir_path, plddt_threshold):
    print(f'len indexes is : {len(indexes)}')
    patches_dict = {}
    all_keys = list(all_predictions['dict_resids'].keys())[indexes[i]:indexes[i + 1]]
    cnt = 0
    for key in all_keys:
        print("i= ", i, " cnt = ", cnt, " key = ", key)
        cnt += 1
        try:
            patches_dict[key] = Protein(key, plddt_threshold)
        except SizeDifferentiationException as e:
            print(e)
            continue
        except Exception as e:
            print(e)
            continue
    save_as_pickle(patches_dict, os.path.join(os.path.join(dir_path, 'proteinObjectsWithEvoluion' + str(i))))


def make_dict_with_integration_keys(all_predictions):
    all_predictions_2d = load_as_pickle(os.path.join(ubdPath, os.path.join('Predictions', 'all_predictions_0310.pkl')))
    keys = all_predictions_2d['dict_sources'].keys()
    for dictKey in all_predictions.keys():
        all_predictions[dictKey] = {key: all_predictions[dictKey][key] for key in all_predictions[dictKey].keys() if
                                    key in keys}


def pkl_components_and_source():
    i = sys.argv[1]
    objs = load_as_pickle(
        os.path.join(ubdPath, os.path.join('newListOfProteinObjects', 'newlistOfProteinObjectsForAggregateFunc' + str(
            i) + '.pkl')))
    tuples = [(obj.source, obj.uniprot_name, obj.connected_components_tuples) for obj in objs]
    save_as_pickle(tuples, os.path.join(ubdPath, os.path.join('newProteinConnectedComponents',
                                                              'newProteinConnectedComponents' + str(
                                                                  i))))


# pklComponentsAndSource()
def repeating_uniprots_to_filter():
    # Read the CSV file into a DataFrame
    df = pd.read_csv(os.path.join(ubdPath, os.path.join('protein_classification', 'uniprotnamecsCSV.csv')))
    # Replace 'your_file.csv' with the actual file path
    # Get unique values from 'proteome' column
    unique_proteome_values = df['proteome'].unique()
    # Find values from 'proteome' that appear in at least one more column
    common_values = set()
    for column in df.columns:
        if column != 'proteome':
            common_values.update(set(unique_proteome_values) & set(df[column]))
    # Display values from 'proteome' that appear in at least one more column
    return list(common_values)


def create_labels_for_components(all_components):
    return np.array([0 if component[0] in NEGATIVE_SOURCES else 1 for component in all_components])


def pkl_labels(all_components, dir_path):
    labels = create_labels_for_components(all_components)
    labels_dir = os.path.join(dir_path, 'labels')
    try:
        os.mkdir(labels_dir)
    except Exception as e:
        print(e)
    save_as_pickle(labels, os.path.join(labels_dir, 'labels'))


def train_k_bin_descretizier_model(data, n_bins_parameter):
    est = KBinsDiscretizer(n_bins=n_bins_parameter, encode='ordinal', strategy='quantile', subsample=None)
    est.fit(data)
    return est


def create_vectorized_data(kBinModel, all_tuples_lists, n_bins_parameter):
    matrix_data = [np.zeros([n_bins_parameter, n_bins_parameter]) for _ in range(len(all_tuples_lists))]
    for i in range(len(all_tuples_lists)):
        for tup in all_tuples_lists[i]:
            input_array = np.array(tup).reshape(1, -1)
            integer_encoding = kBinModel.transform(input_array)
            matrix_data[i][int(integer_encoding[0][0])][int(integer_encoding[0][1])] += 1
    vectorized_data = np.vstack([matrix.flatten() for matrix in matrix_data])
    return vectorized_data


def train_logistic_regression_model(X, Y, class_weights=None):
    model = LogisticRegression(class_weight=class_weights)
    model.fit(X, Y)
    return model


def test_logistic_regression_model(model, X, Y):
    predictions = model.predict(X)
    accuracy = accuracy_score(Y, predictions)
    print(classification_report(Y, predictions))


from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve


def plot_roc(model, labels, data):
    y_probs = model.predict_proba(data)[:, 1]
    fpr, tpr, thresholds = roc_curve(labels, y_probs)
    auc = roc_auc_score(labels, y_probs)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label='ROC Curve (AUC = {:.2f})'.format(auc))
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_precision_recall(y_probs, labels):
    precision, recall, thresholds = precision_recall_curve(labels, y_probs)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label='Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    plt.show()


def prediction_function_using_bayes_factor_computation(logisticPrediction, priorUb, trainingUbRatio):
    # K = utils.KComputation(logisticPrediction, trainingUbRatio)
    K = None
    finalPrediction = (K * priorUb) / ((K * priorUb) + (1 - priorUb))
    return finalPrediction


def update_function(probabilities, priorUb, trainingUbRatio):
    updated_probability = prediction_function_using_bayes_factor_computation(probabilities[1], priorUb, trainingUbRatio)
    probabilities[1] = updated_probability
    probabilities[0] = 1 - updated_probability


def pkl_components_out_of_protein_objects(dirPath):
    list_of_protein_lists = [load_as_pickle(
        os.path.join(dirPath, 'proteinObjectsWithEvoluion' + str(i) + '.pkl')) for i in
        range(len(indexes) - 1)]
    concatenated_list_of_proteins = [protein for sublist in list_of_protein_lists for protein in sublist]
    all_components4d = [(protein.source, protein.uniprot_name, protein.connected_components_tuples, protein.size,
                         len(protein.connected_components_tuples)) for protein in concatenated_list_of_proteins]
    components_dir = os.path.join(dirPath, 'components')
    try:
        os.mkdir(components_dir)
    except Exception as e:
        print(e)
    save_as_pickle(all_components4d, os.path.join(components_dir, 'components'))
    return all_components4d


def create_combined_csv(gridSearchDir, dirName, gridSearchDir2, dirName2, plddtThreshold, plddtThreshold2):
    # Read the first CSV file
    df1 = pd.read_csv(os.path.join(gridSearchDir, 'results_' + dirName + '.csv'))

    # Read the second CSV file
    df2 = pd.read_csv(os.path.join(gridSearchDir2, 'results_' + dirName2 + '.csv'))

    # Merge the two dataframes based on common columns
    merged_df = pd.merge(df1, df2, on=["Entry", "type", "Protein Name", "Organism"])
    merged_df['average Inference'] = (merged_df['Inference Prediction 0.05 prior_x'] + merged_df[
        'Inference Prediction 0.05 prior_y']) / 2
    # Select the desired columns for the new CSV file

    # Rename the columns to differentiate between the two CSV files
    merged_df.rename(
        columns={"Inference Prediction 0.05 prior_x": "Inference Prediction 0.05 prior_" + str(plddtThreshold),
                 "log10Kvalue_x": "log10Kvalue_" + str(plddtThreshold),
                 "Inference Prediction 0.05 prior_y": "Inference Prediction 0.05 prior_" + str(plddtThreshold2),
                 "log10Kvalue_y": "log10Kvalue_" + str(plddtThreshold2)}, inplace=True)

    # Write the merged dataframe to a new CSV file
    path = None
    merged_df.to_csv(
        os.path.join(path.aggregateFunctionMLPDir, "combined_csv_" + str(len(df1['Entry'])) + '.csv'), index=False)


def create_pr_plot_from_results(grid_search_dir):
    utils = None
    predictions, labels, best_architecture = utils.getLabelsPredictionsAndArchitectureOfBestArchitecture(
        grid_search_dir)
    labels = np.array(labels)
    precision, recall, thresholds = precision_recall_curve(labels, predictions)
    sorted_indices = np.argsort(recall)
    sorted_precision = precision[sorted_indices]
    sorted_recall = recall[sorted_indices]
    aucScore = auc(sorted_recall, sorted_precision)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label='Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve, architecture = ' + str(best_architecture) + " auc=" + str(aucScore))
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(grid_search_dir, 'PR plot'))
    plt.close()


def create_log_bayes_distribution_plot_from_results(gridSearchDir):
    utils = None
    predictions, labels, bestArchitecture = utils.getLabelsPredictionsAndArchitectureOfBestArchitecture(gridSearchDir)
    allLog10Kvalues = [np.log10(utils.KComputation(prediction, 0.05)) for prediction in predictions]
    plt.hist(allLog10Kvalues)
    plt.title('logKvalues Distribution, architecture = ' + str(bestArchitecture))
    plt.savefig(os.path.join(gridSearchDir, 'logKvalues Distribution'))
    plt.close()


def plot_plddt_histogram_for_keys(keys, allPredictions, header):
    avgs = []
    for uniprot in keys:
        print(uniprot)
        structurePath = allPredictions['dict_pdb_files'][uniprot]
        structure = parser.get_structure(uniprot, structurePath)
        model = structure.child_list[0]
        assert (len(model) == 1)
        for chain in model:
            residues = aa_out_of_chain(chain)
            avg = np.mean(np.array([residues[i].child_list[0].bfactor for i in range(len(residues))]))
            avgs.append(avg)
        sns.histplot(avgs, kde=True)
        plt.title(header)
        plt.show()


def plot_plddt_histogram_for_positivie_and_proteome(allPredictions):
    keys = allPredictions['dict_sources'].keys()
    positiveKeys = [key for key in keys if
                    allPredictions['dict_sources'][key] in ['E1', 'E2', 'E3', 'ubiquitinBinding', 'DUB']][:50]
    proteomeKeys = [key for key in keys if allPredictions['dict_sources'][key] == 'Human proteome'][:50]
    plot_plddt_histogram_for_keys(positiveKeys, allPredictions, 'Positives plddt histogram')
    plot_plddt_histogram_for_keys(proteomeKeys, allPredictions, 'Proteome plddt histogram')


def create_dummy_pr_plot(dirPath, predictions, labels, header):
    labels = np.array(labels)
    precision, recall, thresholds = precision_recall_curve(labels, predictions)
    sorted_indices = np.argsort(recall)
    sorted_precision = precision[sorted_indices]
    sorted_recall = recall[sorted_indices]
    aucScore = auc(sorted_recall, sorted_precision)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'Precision Recall curve (auc = {aucScore:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(header)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(dirPath, header))
    plt.close()


def plot_dummy_prauc(allPredictions):
    dict_sources = allPredictions['dict_sources']
    dict_predictions_ubiquitin = allPredictions['dict_predictions_ubiquitin']
    labels = np.array([0 if dict_sources[key] in NEGATIVE_SOURCES else 1 for key in dict_sources.keys()])
    predictions = np.array([np.max(dict_predictions_ubiquitin[key]) for key in dict_sources.keys()])
    create_dummy_pr_plot(trainingDataDir, predictions, labels, 'Highest Predicted Amino Acid Baseline')


# common_values = repeatingUniprotsToFilter()
# # existingUniprotNames = [obj.uniprotName for obj in concatenatedListOfProteins]
# for p in concatenatedListOfProteins:
#     if p.uniprotName in common_values:
#         p.source = 'proteome'
#
#
# # missingUniprotsNames = [key for key in allPredictionsUbiq.keys() if key not in uniprotNames]
#
# allComponents3d = [(protein.source, protein.uniprotName, protein.connectedComponentsTuples, protein.size,
#                     len(protein.connectedComponentsTuples)) for protein in concatenatedListOfProteins]
# # allComponents3dFiltered = [component for component in allComponents3d if component[1] not in common_values]
#
# saveAsPickle(allComponents3d,
#              os.path.join(ubdPath, os.path.join('aggregateFunctionMLP', 'allTuplesListsOfLen3_23_3')))

# allComponents3d = loadPickle(
#     os.path.join(ubdPath, os.path.join('aggregateFunctionMLP', 'allTuplesListsOfLen3_23_3.pkl')))
# labels = loadPickle(
#     os.path.join(r'C:\Users\omriy\UBDAndScanNet\newUBD\UBDModel\aggregateFunctionMLP', 'labels3d_23_3.pkl'))


# KBINS
# # n_bins_parameter = 30  # it will actualli be 30^(number of parameter which is 2 because of len(size,average)
# allComponents3dFiltered = loadPickle(
#     os.path.join(ubdPath, os.path.join('aggregateFunctionMLP', 'allTuplesListsOfLen3.pkl')))
# labels = createLabelsForComponents(allComponents3dFiltered)
# saveAsPickle(labels, os.path.join(r'C:\Users\omriy\UBDAndScanNet\newUBD\UBDModel\aggregateFunctionMLP', 'labels3d'))


# print(sum(labels))
# kBinModel = trainKBinDescretizierModel(concatenated_tuples, n_bins_parameter)
# vectorizedData = createVectorizedData(kBinModel, allTuplesLists, n_bins_parameter)
# logisticRegressionModel = trainLogisticRegressionModel(vectorizedData, labels)
# logisticRegressionModelBalanced = trainLogisticRegressionModel(vectorizedData, labels, 'balanced')
# testLogisticRegressionModel(logisticRegressionModel, vectorizedData, labels)
# testLogisticRegressionModel(logisticRegressionModelBalanced, vectorizedData, labels)
# # plt.matshow(logisticRegressionModel.coef_.reshape([30,30]),vmin=-1.,vmax=1,cmap='jet'); plt.colorbar(); plt.show()
# trainingRatio = sum(labels) / len(allTuplesLists)
# ubProbabillits = np.array([row[1] for row in logisticRegressionModel.predict_proba(vectorizedData)])
# finalOutputsTen = [predictionFunctionUsingBayesFactorComputation(proba, 0.1, trainingRatio) for proba in ubProbabillits]
# finalOutputsFifty = [predictionFunctionUsingBayesFactorComputation(proba, 0.5, trainingRatio) for proba in
#                      ubProbabillits]
# KValues = [KComputation(proba, trainingRatio) for proba in ubProbabillits]
# import csv

def readDataFromUni(fileName):
    data_dict = {}
    # Read the TSV file and populate the dictionary
    with open(fileName,
              'r') as tsv_file:
        tsv_reader = csv.reader(tsv_file, delimiter='\t')
        header = next(tsv_reader)  # Get column headers
        for row in tsv_reader:
            key = row[0]  # Use the first column as the key
            row_data = dict(
                zip(header[1:], row[1:]))  # Create a dictionary for the row data (excluding the first column)
            data_dict[key] = row_data
        return data_dict


# data_dict = readDataFromUni(
#     r'C:\Users\omriy\UBDAndScanNet\newUBD\UBDModel\GO\idmapping_2023_12_26.tsv\idmapping_2023_12_26.tsv')

def createInfoCsvLogisticRegression(data_dict, predBayes10, predBayes50, KValues):
    finalOutputsTen = None
    allComponentsFiltered = None

    myList = []
    for i in range(len(finalOutputsTen)):
        uniDict = data_dict[allComponentsFiltered[i][1]]
        myList.append(
            (uniDict['Entry'], uniDict['Protein names'], uniDict['Organism'], predBayes10[i], predBayes50[i],
             KValues[i]))
    headers = ('Entry', 'Protein Name', 'Organism', 'Bayes Prediction 0.1 prior', 'Bayes Prediction 0.5 prior',
               'K value')
    # Define file path for writing
    file_path = 'InfoFileScoringFunction10And50.csv'
    # Write the data to a TSV file
    with open(file_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        # Write headers
        csv_writer.writerow(headers)

        # Write rows of data
        for row in myList:  # Skip the first row since it contains headers
            csv_writer.writerow(row)


# createInfoCsv(data_dict, finalOutputsTen, finalOutputsFifty, KValues)

def getNBiggestFP(labels, predictions, allComponentsFiltered, N):
    negativeIndexes = [i for i in range(len(labels)) if labels[i] == 0]
    topPredictionsIndexes = sorted(negativeIndexes, key=lambda i: predictions[i], reverse=True)[:N]
    NBiggestFP = [(allComponentsFiltered[topPredictionsIndexes[i]][1], predictions[topPredictionsIndexes[i]]) for i in
                  range(N)]
    return NBiggestFP

# NBiggestFP = getNBiggestFP(labels, finalOutputsFifty, allComponentsFiltered, 10)
