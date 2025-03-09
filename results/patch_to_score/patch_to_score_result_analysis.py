import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import csv
from data_preparation.ScanNet.db_creation_scanNet_utils import load_as_pickle
import paths
import re
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc
import numpy as np
from models.patch_to_score.patch_to_score_MLP_utils import k_computation, prediction_function_using_bayes_factor_computation

import math
import requests
import time

def create_aucs_csv(models_dir,results_dir):
    aucs_data = []
    
    for filename in os.listdir(models_dir):
        match = re.match(r'totalAucs:n_layers:(\d+)_m_a:(\d+)_m_c:(\d+).pkl', filename)
        if match:
            filepath = os.path.join(models_dir, filename)
            tuples_list = load_as_pickle(filepath)
            for ((m_a, m_b, m_c, n_layers, n_early_stopping_epochs, batch_size), pr_auc) in tuples_list:
                aucs_data.append({
                    'n_layers': n_layers,
                    'm_a': m_a,
                    'm_b': m_b,
                    'm_c': m_c,
                    'n_early_stopping_epochs': n_early_stopping_epochs,
                    'batch_size': batch_size,
                    'auc': pr_auc
                })
    
    # Sorting by 'auc' in descending order
    aucs_data.sort(key=lambda x: x['auc'], reverse=True)
    
    # Writing to CSV
    with open(os.path.join(results_dir,'aucs_data.csv'), 'w', newline='') as csvfile:
        field_names = ['n_layers', 'm_a', 'm_b', 'm_c', 'n_early_stopping_epochs', 'batch_size', 'auc']
        writer = csv.DictWriter(csvfile, fieldnames=field_names)
        writer.writeheader()
        for data in aucs_data:
            writer.writerow(data)

def get_best_architecture_parts(results_dir):
    # Read the CSV file
    data = pd.read_csv(os.path.join(results_dir, 'aucs_data.csv'))
    
    # Find the row with the highest AUC score
    best_row = data.loc[data['auc'].idxmax()]
    
    # Extract the architecture parts
    n_layers = str(int(best_row['n_layers']))
    m_a = str(int(best_row['m_a']))
    m_b = str(int(best_row['m_b']))
    m_c = str(int(best_row['m_c']))    
    return n_layers, m_a, m_b, m_c

def get_best_architecture_models_path(models_dir,results_dir):
    # Assuming get_best_architecture_parts returns the best architecture parts
    n_layers, m_a, m_b, m_c = get_best_architecture_parts(results_dir)
    
    # Construct the directory name based on the architecture parts
    best_architecture_dir = f'architecture:{n_layers}_{m_a}_{m_b}_{m_c}'
    
    # Return the full path to the best architecture directory
    return os.path.join(models_dir, best_architecture_dir)

def save_pr_plot(architecture_models_dir,results_architecture_dir):
    # Load predictions and labels
    predictions = np.load(os.path.join(architecture_models_dir,"predictions_test.npy"))
    labels = np.load(os.path.join(architecture_models_dir,"labels_test.npy"))
    
    # Calculate Precision-Recall and AUC
    precision, recall, _ = precision_recall_curve(labels, predictions)
    pr_auc = auc(recall, precision)
    
    # Plot PR curve
    plt.figure()
    plt.plot(recall, precision, label=f'PR curve (area = {pr_auc:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.legend(loc="best")
    if not os.path.exists(results_architecture_dir):
        os.makedirs(results_architecture_dir)
    # Save the plot
    plt.savefig(os.path.join(results_architecture_dir,"pr_plot_with_auc.png"))

def create_log_bayes_distribution_plot_from_results(architecture_models_dir,results_architecture_dir):
    plt.clf()
    predictions = np.load(os.path.join(architecture_models_dir,"predictions_test.npy")).squeeze()
    allLog10Kvalues = np.array([np.log10(k_computation(prediction, 0.05)) if k_computation(prediction, 0.05) != None else math.inf for prediction in predictions])
    plt.hist(allLog10Kvalues)
    plt.title('logKvalues Distribution')
    plt.savefig(os.path.join(results_architecture_dir, 'logKvalues Distribution'))
    plt.close()

def createYhatGroupsFromPredictions(predictions, dictsForTraining, testOn='cv'):
    yhat_groups = []
    cnt = 0
    for i in range(len(dictsForTraining)):
        y = 'y_' + testOn
        yhat_groups.append([predictions[i] for i in range(cnt, cnt + dictsForTraining[i][y].size)])
        cnt += dictsForTraining[i][y].size
    return yhat_groups

def createInfoCsv(yhat_groups, dictsForTraining, allInfoDicts, dataDictPath, outputPath, testOn='cv'):
    data_dict = load_as_pickle(dataDictPath)
    allKvalues = []
    for i in range(len(dictsForTraining)):
        trainingUbRatio = np.mean(dictsForTraining[i]['y_train'])
        allKvalues.extend([k_computation(yhat_groups[i][j], trainingUbRatio) for j in range(len(yhat_groups[i]))])
    Inference5PercentPredictions = [prediction_function_using_bayes_factor_computation(0.05, KValue) for KValue in
                                    allKvalues]
    logKValues = [np.log10(k) for k in allKvalues]
    uniDictList = []
    myList = []
    types = []
    for i in range(len(allInfoDicts)):
        x = 'x_' + testOn
        for j in range(len(allInfoDicts[i][x])):
            if allInfoDicts[i][x][j][1] in data_dict:
                uniDict = data_dict[allInfoDicts[i][x][j][1]]
            else:
                uniDict = {'Entry': allInfoDicts[i][x][j][1], 'Protein names': 'No Info', 'Organism': 'No Info'}
            uniDictList.append(uniDict)
            types.append(allInfoDicts[i][x][j][0])
    assert len(allKvalues) == len(uniDictList)
    for i in range(len(allKvalues)):
        uniDict = uniDictList[i]
        myList.append(
            (uniDict['Entry'], types[i], uniDict['Protein names'], uniDict['Organism'], Inference5PercentPredictions[i],
             logKValues[i]))

    headers = ('Entry', 'type', 'Protein Name', 'Organism', 'Inference Prediction 0.05 prior',
               'log10Kvalue')
    # Write the data to a TSV file
    with open(outputPath, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        # Write headers
        csv_writer.writerow(headers)

        # Write rows of data
        for row in myList:  # Skip the first row since it contains headers
            csv_writer.writerow(row)

def createCSVFileFromResults(gridSearchDir, trainingDictsDir, dir_path):
    predictions = np.load(os.path.join(dir_path, 'predictions.npy'))
    allInfoDicts = load_as_pickle(os.path.join('asf', 'allInfoDicts.pkl'))
    dictsForTraining = load_as_pickle(os.path.join(trainingDictsDir, 'dictsForTraining.pkl'))
    data_dict_path = os.path.join(paths.patch_to_score_dataset_path, 'AllOrganizemsDataDict.pkl')
    yhat_groups = createYhatGroupsFromPredictions(predictions, dictsForTraining)
    createInfoCsv(yhat_groups, dictsForTraining, allInfoDicts, data_dict_path, os.path.join(dir_path, 'InferenceResults.csv'))
    
def create_best_architecture_results_dir(best_architecture_models_path,results_dir):
    best_architecture_results_dir = os.path.basename(best_architecture_models_path)
    best_architecture_results_dir = os.path.join(results_dir, best_architecture_results_dir)
    os.makedirs(best_architecture_results_dir, exist_ok=True)
    return best_architecture_results_dir


if __name__ == "__main__":
    # create_aucs_csv(paths.with_MSA_50_plddt_0304_models_dir,paths.with_MSA_50_plddt_0304_results_dir)
    best_architecture_models_path = get_best_architecture_models_path(paths.with_MSA_50_plddt_0304_models_dir,paths.with_MSA_50_plddt_0304_results_dir)
    best_architecture_results_dir = create_best_architecture_results_dir(best_architecture_models_path,paths.with_MSA_50_plddt_0304_results_dir)
    save_pr_plot(best_architecture_models_path,best_architecture_results_dir)
    create_log_bayes_distribution_plot_from_results(best_architecture_models_path,best_architecture_results_dir)
