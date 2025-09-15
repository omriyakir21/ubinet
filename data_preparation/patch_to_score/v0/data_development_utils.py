import sys
import os
import csv
from data_preparation.ScanNet.db_creation_scanNet_utils import save_as_pickle, load_as_pickle, \
    aa_out_of_chain, get_str_seq_of_chain
import networkx as nx
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from matplotlib import pyplot as plt
import pandas as pd
from Bio.PDB import MMCIFParser, PDBParser
import seaborn as sns
from sklearn.metrics import auc
import paths
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np
import tensorflow as tf
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.PDBIO import PDBIO

MAX_NUMBER_OF_COMPONENTS = 10
NEGATIVE_SOURCES = set(
    ['Yeast proteome', 'Human proteome', 'Ecoli proteome', 'Celegans proteome', 'Arabidopsis proteome'])

POSITIVE_SOURCES = set(['E1', 'E2', 'E3', 'ubiquitinBinding', 'DUB'])

parser = PDBParser()
parserMMcif = MMCIFParser()

server_PDBs = True
DISTANCE_THRESHOLD = 10

# to be removed
trainingDataDir = None
ubdPath = None


class SizeDifferentiationException(Exception):
    def __init__(self, uniprotName):
        super().__init__("uniprotName: ", uniprotName, "\n")


class Protein:
    def __init__(self, uniprot_name, plddt_threshold, all_predictions,percentile_90,with_pesto):
        self.uniprot_name = uniprot_name
        self.percentile_90 = percentile_90 
        self.predictions_dict = self.fill_predictions_dict(all_predictions,with_pesto)
        self.source = all_predictions['dict_sources'][uniprot_name]
        self.plddt_values = self.get_plddt_values()
        self.sequence = self.get_sequence()
        self.size = len(self.sequence)
        self.graph = nx.Graph()
        self.create_graph(plddt_threshold)
        self.connected_components_tuples = self.create_connected_components_tuples()

    def fill_predictions_dict(self,all_predictions,with_pesto):
        predictions_dict = {}
        predictions_dict['scanNet_ubiq'] = all_predictions['dict_predictions_ubiquitin'][self.uniprot_name]
        predictions_dict['scanNet_protein'] = all_predictions['dict_predictions_interface'][self.uniprot_name]
        if with_pesto:
            predictions_dict['pesto_protein'] = all_predictions['pesto_protein'][self.uniprot_name]
            predictions_dict['pesto_dna_rna'] = all_predictions['pesto_dna_rna'][self.uniprot_name]
            predictions_dict['pesto_ion'] = all_predictions['pesto_ion'][self.uniprot_name]
            predictions_dict['pesto_ligand'] = all_predictions['pesto_ligand'][self.uniprot_name]
            predictions_dict['pesto_lipid'] = all_predictions['pesto_lipid'][self.uniprot_name]
        return predictions_dict

        

    def get_residues(self):
        structure = self.get_structure()
        model = structure.child_list[0]
        assert (len(model) == 1)
        for chain in model:
            return aa_out_of_chain(chain)
    
    def get_structure(self, path=None):
        if path is not None:
            structurePath = path
        else:
            if self.source in NEGATIVE_SOURCES:
                structurePath = os.path.join(paths.AFDB_source_patch_to_score_path, self.source.split(" ")[0],
                                             self.uniprot_name + '.pdb')  # the name of the AFDB dirs doesnt end with proteome thats the reason of the split
            else:
                structurePath = os.path.join(paths.GO_source_patch_to_score_path, self.source,
                                             self.uniprot_name + '.pdb')
        # print(structurePath)
        if not os.path.exists(structurePath):
            # print(f"path does not exist for : {self.uniprot_name}")
            # structurePath2 = all_predictions['dict_pdb_files'][self.uniprot_name]
            # convert_cif_to_pdb(structurePath2, structurePath)
            # print(f"created new path in : {structurePath}")
            raise Exception("path does not exist")
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
        if len(residues) != len(self.predictions_dict['scanNet_ubiq']):
            raise SizeDifferentiationException(self.uniprot_name)
        for i in range(len(residues)):
            plddtVal = residues[i].child_list[0].bfactor
            # print(f'plddt_val {plddtVal}')
            # print(f'threshold {plddt_threshold}')
            # print(f' self.predictions_dict[scanNet_ubiq][i] {self.predictions_dict["scanNet_ubiq"][i]}')
            # print(f'self.percentile_90 {self.percentile_90}')
            if plddtVal > plddt_threshold and self.predictions_dict['scanNet_ubiq'][i] > self.percentile_90:
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

    def get_sequence(self):
        structure = self.get_structure()
        model = structure.child_list[0]
        assert (len(model) == 1)
        for chain in model:
            seq = get_str_seq_of_chain(chain)
        return seq

    def create_graph(self, plddt_threshold):
        structure = self.get_structure()
        model = structure.child_list[0]
        assert (len(model) == 1)
        for chain in model:
            residues = aa_out_of_chain(chain)
            nodes = self.create_nodes_for_graph(residues, plddt_threshold)
            valid_residues = [residues[i] for i in nodes]
            edges = self.create_edges_for_graph(valid_residues, nodes)
            self.graph.add_nodes_from(nodes)
            self.graph.add_edges_from(edges)

    def create_connected_components_tuples(self):
        tuples = []
        connected_components = list(nx.connected_components(self.graph))
        for component_set in connected_components:
            # average_ubiq, average_non_ubiq, average_plddt = self.calculate_average_predictions_for_component(
                # component_set)
            patch_dict = self.calculate_average_predictions_for_component(component_set)
            patch_size = len(component_set)
            tuples.append((patch_size, patch_dict, list(component_set)))
        return tuples

    def calculate_average_predictions_for_component(self, indexSet):
        patch_dict = {}
        indexes = list(indexSet)
        plddt_values = [self.plddt_values[index] for index in indexes]
        for key,val in self.predictions_dict.items():
            patch_dict[f'average_{key}'] = sum([val[index] for index in indexes]) / len(indexes)
        patch_dict['average_plddt'] = sum(plddt_values) / len(plddt_values)
        return patch_dict


def create_90_percentile(all_predictions: dict, percentile_90_path: str):
    all_predictions_ubiq = all_predictions['dict_predictions_ubiquitin']
    all_predictions_ubiq_flatten = [value for values_list in all_predictions_ubiq.values() for value in values_list]
    percentile_90 = np.percentile(all_predictions_ubiq_flatten, 90)
    print(percentile_90)
    save_as_pickle(percentile_90,percentile_90_path)

def c_alpha_distance(atom1, atom2):
    vector1 = atom1.get_coord()
    vector2 = atom2.get_coord()
    distance = np.sqrt(((vector2[np.newaxis] - vector1[np.newaxis]) ** 2).sum(-1))
    return distance


def create_patches_dict(i, dir_path, plddt_threshold, all_predictions,percentile_90,with_pesto):
    indexes_path = os.path.join(dir_path, 'indexes.pkl')
    if not os.path.exists(indexes_path):
        indexes = list(range(0, len(all_predictions['dict_resids']) + 1, 1500)) + [len(all_predictions['dict_resids'])]
        save_as_pickle(indexes, indexes_path)
    indexes = load_as_pickle(indexes_path)
    print(f'len indexes is : {len(indexes)}')
    patches_dict = {}
    all_keys = list(all_predictions['dict_resids'].keys())[indexes[i]:indexes[i + 1]]
    cnt = 0
    for key in all_keys:
        print("i= ", i, " cnt = ", cnt, " key = ", key)
        cnt += 1
        patches_dict[key] = Protein(key, plddt_threshold, all_predictions, percentile_90,with_pesto)
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
    """
    Plot the PRAUC of the dummy model (protein prediction is the prediction of the highest amino acid prediction)
    """
    dict_sources = allPredictions['dict_sources']
    dict_predictions_ubiquitin = allPredictions['dict_predictions_ubiquitin']
    labels = np.array([0 if dict_sources[key] in NEGATIVE_SOURCES else 1 for key in dict_sources.keys()])
    predictions = np.array([np.max(dict_predictions_ubiquitin[key]) for key in dict_sources.keys()])
    create_dummy_pr_plot(trainingDataDir, predictions, labels, 'Highest Predicted Amino Acid Baseline')


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

 
def extract_protein_data(proteins, max_number_of_components):
    data_components_flattend = []
    data_protein_size = []
    data_number_of_components = []
    data_components = []
    for protein in proteins:
        # Sort components by average_ubiq in descending order and take the top 10
        top_components = sorted(protein.connected_components_tuples, key=lambda x: x[1]['average_scanNet_ubiq'], reverse=True)[
                         :max_number_of_components]
        data_components.append([component[:2] for component in top_components])
        for component in top_components:
            patch_size, patch_dict = component[:2]
            patch_dict_flattened = [float(val) for val in patch_dict.values()]
            patch_flattened = [patch_size] + patch_dict_flattened
            print(f'patch_dict : {patch_dict}')
            print(f'patch_flattened : {patch_flattened}')
            data_components_flattend.append(patch_flattened)
        data_protein_size.append(protein.size)
        data_number_of_components.append(len(top_components))
    return data_components_flattend, data_protein_size, data_number_of_components, data_components


def fit_protein_data(all_data_components, all_data_protein_size, all_data_number_of_components, dir_path,
                     max_number_of_components):
    # Fit the scalers
    scaler_size = StandardScaler()
    scaler_components = StandardScaler()
    scaler_size.fit(all_data_protein_size.reshape(-1, 1))
    scaler_components.fit(all_data_components)
 
    # Fit the encoder
    encoder = OneHotEncoder(sparse_output=False, categories=[range(max_number_of_components + 1)])
    encoder.fit(all_data_number_of_components.reshape(-1, 1))

    # Save the scalers and encoder
    save_as_pickle(scaler_size, os.path.join(dir_path, 'scaler_size.pkl'))
    save_as_pickle(scaler_components, os.path.join(dir_path, 'scaler_components.pkl'))
    save_as_pickle(encoder, os.path.join(dir_path, 'encoder.pkl'))


def transform_protein_data(protein, scaler_size, scaler_components, encoder, max_number_of_components,with_pesto):
    scaled_size = scaler_size.transform(np.array([protein.size]).reshape(-1, 1))

    # Extract, scale, and pad the components
    top_components = sorted(protein.connected_components_tuples, key=lambda x: x[1]['average_scanNet_ubiq'], reverse=True)[
                     :max_number_of_components]
    component_size = 9 if with_pesto else 4
    if len(top_components) == 0:
        protein_components_scaled = np.zeros((0, component_size))
    else:
        protein_components_flattend = []
        for component in top_components:
            patch_size, patch_dict = component[:2]
            patch_dict_flattened = [float(val) for val in patch_dict.values()]
            patch_flattened = [patch_size] + patch_dict_flattened
            protein_components_flattend.append(patch_flattened)
        protein_components_flattend = np.array(protein_components_flattend)
        protein_components_scaled = scaler_components.transform(protein_components_flattend)
    if len(protein_components_scaled) < max_number_of_components:
        padding = ((0, max_number_of_components - len(protein_components_scaled)), (0, 0))
        protein_components_scaled = np.pad(protein_components_scaled, padding, mode='constant', constant_values=0)

    # Encode the number of components
    encoded_components = encoder.transform(np.array([len(top_components)]).reshape(-1, 1))

    # Convert to tensors
    scaled_size_tensor = tf.convert_to_tensor(scaled_size)
    scaled_components_tensor = tf.convert_to_tensor(protein_components_scaled)
    encoded_components_tensor = tf.convert_to_tensor(encoded_components)

    return scaled_size_tensor, scaled_components_tensor, encoded_components_tensor


def transform_protein_data_list(proteins, scaler_size_path, scaler_components_path, encoder_path,
                                max_number_of_components,with_pesto):
    scaler_size = load_as_pickle(scaler_size_path)
    scaler_components = load_as_pickle(scaler_components_path)
    encoder = load_as_pickle(encoder_path)
    scaled_sizes = []
    scaled_components_list = []
    encoded_components_list = []

    for protein in proteins:
        scaled_size_tensor, scaled_components_tensor, encoded_components_tensor = transform_protein_data(
            protein, scaler_size, scaler_components, encoder, max_number_of_components,with_pesto)

        scaled_sizes.append(scaled_size_tensor)
        scaled_components_list.append(scaled_components_tensor)
        encoded_components_list.append(encoded_components_tensor)

    # Print shapes of lists before stacking
    print("scaled_sizes list shape:", len(scaled_sizes), scaled_sizes[0].shape if scaled_sizes else None)
    print(f'scaled_sizes is {scaled_sizes}')
    print("scaled_components_list shape:", len(scaled_components_list),
          scaled_components_list[0].shape if scaled_components_list else None)
    print("encoded_components_list shape:", len(encoded_components_list),
          encoded_components_list[0].shape if encoded_components_list else None)
    scaled_sizes = tf.squeeze(tf.convert_to_tensor(scaled_sizes),axis=1)
    encoded_components_list = tf.squeeze(tf.convert_to_tensor(encoded_components_list),axis=1)
    return scaled_sizes, tf.convert_to_tensor(scaled_components_list), encoded_components_list

def save_as_tensor(data, path):
    tensor = tf.convert_to_tensor(data)
    serialized_tensor = tf.io.serialize_tensor(tensor)
    tf.io.write_file(path, serialized_tensor)

def load_as_tensor(path,out_type=tf.double):
    serialized_tensor = tf.io.read_file(path)
    tensor = tf.io.parse_tensor(serialized_tensor,out_type=out_type)  # Adjust `out_type` as needed
    return tensor

def create_training_folds(groups_indices, scaled_sizes_path, scaled_components_list_path, encoded_components_list_path,
                          all_uniprots_path, labels_path):
    folds_training_dicts = []
    scaled_sizes = load_as_tensor(scaled_sizes_path)
    scaled_components = load_as_tensor(scaled_components_list_path)
    encoded_components = load_as_tensor(encoded_components_list_path)
    labels = load_as_tensor(labels_path,tf.int32)
    uniprots = load_as_pickle(all_uniprots_path)

    for i in range(5):
        training_dict = {}

        training_indices = tf.constant(np.concatenate((groups_indices[(i + 2) % 5] , groups_indices[(i + 3) % 5] , groups_indices[(i + 4) % 5])))
        validation_indices = tf.constant(groups_indices[i])
        test_indices = tf.constant(groups_indices[(i + 1) % 5])

        # When using these indices to index tensors
        training_dict['sizes_train'] = tf.gather(scaled_sizes, training_indices)
        training_dict['components_train'] = tf.gather(scaled_components, training_indices)
        training_dict['num_patches_train'] = tf.gather(encoded_components, training_indices)
        training_dict['uniprots_train'] = [uniprots[i] for i in training_indices.numpy()]  # Assuming uniprots is a list and not a tensor
        training_dict['labels_train'] = tf.gather(labels, training_indices)
        training_dict['sizes_validation'] = tf.gather(scaled_sizes, validation_indices)
        training_dict['components_validation'] = tf.gather(scaled_components, validation_indices)
        training_dict['num_patches_validation'] = tf.gather(encoded_components, validation_indices)
        training_dict['uniprots_validation'] = [uniprots[i] for i in validation_indices.numpy()]  # Assuming uniprots is a list and not a tensor
        training_dict['labels_validation'] = tf.gather(labels, validation_indices)
        training_dict['sizes_test'] = tf.gather(scaled_sizes, test_indices)
        training_dict['components_test'] = tf.gather(scaled_components, test_indices)
        training_dict['num_patches_test'] = tf.gather(encoded_components, test_indices)
        training_dict['uniprots_test'] = [uniprots[i] for i in test_indices.numpy()]  # Assuming uniprots is a list and not a tensor
        training_dict['labels_test'] = tf.gather(labels, test_indices)
 
        folds_training_dicts.append(training_dict)
    return folds_training_dicts

def convert_cif_to_pdb(cif_path, pdb_path):
    # Parse the .cif file to get the structure
    parser = MMCIFParser()
    structure = parser.get_structure("structure_id", cif_path)
    
    # Initialize PDBIO object
    io = PDBIO()
    
    # Set the structure for the PDBIO object
    io.set_structure(structure)
    
    # Save the structure as a .pdb file
    io.save(pdb_path)
