import math
import os
import sys
import csv
import pickle

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from PIL.ImageOps import pad
from sklearn.utils import compute_class_weight
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, classification_report, roc_auc_score, roc_curve, precision_recall_curve, auc
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import backend
from tensorflow.keras.layers import Dense, GlobalAveragePooling1D, Reshape, Masking

from data_preparation.patch_to_score.v0.data_development_utils import NEGATIVE_SOURCES,POSITIVE_SOURCES
import paths


def plotPrecisionRecall(y_probs, labels):
    precision, recall, thresholds = precision_recall_curve(labels, y_probs)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label='Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    plt.show()


def createRandomDataSet(size):
    # Set the number of matrices (m) in the list
    m_matrices = size  # Adjust as needed

    # Create a list to store the matrices
    matrices_list = []

    # Generate random matrices for each index in the list
    for _ in range(m_matrices):
        k_i = np.random.randint(5, 21)  # Randomly sample k_i between 5 and 20
        matrix_i = np.random.rand(k_i, 3)  # Create a random matrix of dimensions (k_i, 3)
        matrices_list.append(tf.constant(matrix_i))

    # Generate random binary labels (0 or 1) for each vector
    random_labels_np = np.random.randint(2, size=size)

    # Convert NumPy arrays to TensorFlow tensors
    random_labels_tf = tf.reshape(tf.constant(random_labels_np), (-1, 1))
    return matrices_list, random_labels_tf


def sortPatches(x):
    try:

        for i in range(len(x)):
            # Get indices that would sort the tensor along the second column
            if x[i].shape != (0,):
                sorted_indices = tf.argsort(x[i][:, 1])
                # Use tf.gather to rearrange rows based on the sorted indices
                sorted_tensor = tf.gather(x[i], sorted_indices)
                x[i] = sorted_tensor
    except Exception:
        print(1)


def divideTrainValidationTest(x, y):
    # Get 60% of the dataset as the training set. Put the remaining 40% in temporary variables: x_ and y_.
    x_train, x_, y_train, y_ = train_test_split(x, y, test_size=0.40, random_state=1)

    # Split the 40% subset above into two: one half for cross validation and the other for the test set
    x_cv, x_test, y_cv, y_test = train_test_split(x_, y_, test_size=0.50, random_state=1)
    # Delete temporary variables
    del x_, y_
    return x_train, x_cv, x_test, y_train, y_cv, y_test


def padXValues(x_train, x_cv, x_test, maxNumberOfPatches):
    x_train = tf.keras.preprocessing.sequence.pad_sequences(
        x_train, padding="post", maxlen=maxNumberOfPatches, dtype='float32'
    )
    x_cv = tf.keras.preprocessing.sequence.pad_sequences(
        x_cv, padding="post", maxlen=maxNumberOfPatches, dtype='float32'
    )
    x_test = tf.keras.preprocessing.sequence.pad_sequences(
        x_test, padding="post", maxlen=maxNumberOfPatches, dtype='float32'
    )
    return x_train, x_cv, x_test


def Scale2DUtil(x, scalerSize, scalerAverageUbBinding):
    for i in range(len(x)):
        if x[i].shape != (0,):
            size_scaled = scalerSize.transform(tf.reshape(x[i][:, 0], [-1, 1]))
            averages_scaled = scalerAverageUbBinding.transform(tf.reshape(x[i][:, 1], [-1, 1]))
            x[i] = np.concatenate((size_scaled, averages_scaled), axis=1)


def scaleXValues2D(x_train, x_cv, x_test):
    # Scale the features using the z-score
    allTupleSizesTrain = np.concatenate([tuples[:, 0] for tuples in x_train if tuples.shape != (0,)])
    allTupleUbAveragesTrain = np.concatenate([tuples[:, 1] for tuples in x_train if tuples.shape != (0,)])
    scalerSize = StandardScaler()
    scalerAverageUbBinding = StandardScaler()
    scalerSize.fit(allTupleSizesTrain.reshape((-1, 1)))
    scalerAverageUbBinding.fit(allTupleUbAveragesTrain.reshape((-1, 1)))
    Scale2DUtil(x_train, scalerSize, scalerAverageUbBinding)
    Scale2DUtil(x_cv, scalerSize, scalerAverageUbBinding)
    Scale2DUtil(x_test, scalerSize, scalerAverageUbBinding)
    return x_train, x_cv, x_test


def Scale4DUtil(x, scalerSize, scalerAverageUbBinding, plddtScaler):
    for i in range(len(x)):
        if x[i].shape == (0,):
            continue
        size_scaled = scalerSize.transform(tf.reshape(x[i][:, 0], [-1, 1]))
        ubAverages_scaled = scalerAverageUbBinding.transform(tf.reshape(x[i][:, 1], [-1, 1]))
        nonUbAverages_scaled = scalerAverageUbBinding.transform(tf.reshape(x[i][:, 2], [-1, 1]))
        plddt_scaled = plddtScaler.transform(tf.reshape(x[i][:, 3], [-1, 1]))
        x[i] = np.concatenate((size_scaled, ubAverages_scaled, nonUbAverages_scaled, plddt_scaled), axis=1)


def scaleXComponents4D(x_train_components, x_cv_components, x_test_components, modelsDir):
    # Scale the features using the z-score
    allTupleSizesTrain = np.concatenate([tuples[:, 0] for tuples in x_train_components if tuples.shape != (0,)])
    allTupleUbAveragesTrain = np.concatenate([tuples[:, 1] for tuples in x_train_components if tuples.shape != (0,)])
    allTuplePlddtTrain = np.concatenate([tuples[:, 3] for tuples in x_train_components if tuples.shape != (0,)])
    scalerSize = StandardScaler()
    scalerAverageUbBinding = StandardScaler()
    plddtScaler = StandardScaler()
    scalerSize.fit(allTupleSizesTrain.reshape((-1, 1)))
    scalerAverageUbBinding.fit(allTupleUbAveragesTrain.reshape((-1, 1)))
    plddtScaler.fit(allTuplePlddtTrain.reshape((-1, 1)))
    Scale4DUtil(x_train_components, scalerSize, scalerAverageUbBinding, plddtScaler)
    Scale4DUtil(x_cv_components, scalerSize, scalerAverageUbBinding, plddtScaler)
    Scale4DUtil(x_test_components, scalerSize, scalerAverageUbBinding, plddtScaler)
    saveAsPickle(scalerSize, os.path.join(modelsDir, 'sizeComponentScaler'))
    saveAsPickle(scalerAverageUbBinding, os.path.join(modelsDir, 'averageUbBindingScaler'))
    saveAsPickle(plddtScaler, os.path.join(modelsDir, 'plddtScaler'))
    return x_train_components, x_cv_components, x_test_components


def getScaleXSizes3D(x_train_sizes, x_cv_sizes, x_test_sizes, modelsDir):
    # Scale the features using the z-score
    scalerSize = StandardScaler()
    x_train_sizes_scaled = scalerSize.fit_transform(x_train_sizes.reshape((-1, 1)))
    x_cv_sizes_scaled = scalerSize.transform(x_cv_sizes.reshape((-1, 1)))
    x_test_sizes_scaled = scalerSize.transform(x_test_sizes.reshape((-1, 1)))
    saveAsPickle(scalerSize, os.path.join(modelsDir, 'proteinSizeScaler'))
    return x_train_sizes_scaled, x_cv_sizes_scaled, x_test_sizes_scaled


def saveAsPickle(object, fileName):
    with open(fileName + '.pkl', 'wb') as file:
        pickle.dump(object, file)


def loadPickle(fileName):
    with open(fileName, 'rb') as file:
        object = pickle.load(file)
        return object


def build_models():
    tf.random.set_seed(20)

    model_1 = Sequential(
        [tf.keras.Input(shape=(maxNumberOfPatches, 3)),
         Masking(mask_value=0.0),
         Dense(16, activation='relu'),
         GlobalAveragePooling1D(data_format='channels_last'),
         Dense(1, activation='linear')
         ],
        name='model_1'
    )

    model_2 = Sequential(
        [
            tf.keras.Input(shape=(maxNumberOfPatches, 3)),
            Masking(mask_value=0.0),
            Dense(25, activation='relu'),
            Dense(16, activation='relu'),
            GlobalAveragePooling1D(data_format='channels_last'),
            Dense(1, activation='linear')
        ],
        name='model_2'
    )

    model_3 = Sequential(
        [
            tf.keras.Input(shape=(maxNumberOfPatches, 3)),
            Masking(mask_value=0.0),
            Dense(32, activation='relu'),
            Dense(25, activation='relu'),
            Dense(16, activation='relu'),
            GlobalAveragePooling1D(data_format='channels_last'),
            Dense(1, activation='linear')
        ],
        name='model_3'
    )

    model_list = [model_1, model_2, model_3]

    return model_list


def divideXData(x):
    components = [tup[2] for tup in x]
    sizes = np.array([tup[3] for tup in x])
    n_patches = np.array([tup[4] for tup in x])
    # componentWithoutIndexes = [np.array(tup[:4])for tup in components]
    # componentIndexes = [tup[4] for tup in components]

    return components, sizes, n_patches


def plotROC(y_probs, labels):
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


def plot_precision_recall(y_probs, labels, header,save_path):
    precision, recall, _ = precision_recall_curve(labels, y_probs)
    pr_auc = auc(recall, precision)
    plt.figure(figsize=(8, 6))
    pr_auc = round(pr_auc, 3)
    plt.plot(recall, precision, label='Precision-Recall Curve, AUC =' + str(pr_auc))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(header)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def build_model_concat_size_and_n_patches_same_number_of_layers(architecture_dict, input_shape, 
                                                                max_number_of_patches: int):
    '''
    :param m_a: size of the hidden layers in the MLP of the components
    :param m_b: size of the hidden layers in the MLP of the concatenated size and number of patches
    :param m_c: size of the hidden layers in the MLP of the concatenated global sum output and size + n_patches MLP output
    :param n_layers: number of layers in each of the MLPs
    :return:
    '''
    # Define the input shape 
    input_data = tf.keras.Input(shape=input_shape, name='patches_input')
    size_value = tf.keras.Input(shape=(1,), name='extra_value_input')
    n_patches_hot_encoded_value = tf.keras.Input(shape=(max_number_of_patches+ 1,), name='hot_encoded_value_input')
    masked_input = tf.keras.layers.Masking(mask_value=0.0)(input_data)

    currentOutput = masked_input
    for i in range(architecture_dict['n_layers']):
        dense_output = tf.keras.layers.Dense(architecture_dict['m_a'], activation='linear')(currentOutput)
        batchNorm = tf.keras.layers.BatchNormalization(momentum=0.75)(dense_output)
        activation = tf.keras.layers.ReLU()(batchNorm)
        currentOutput = activation

    global_pooling_output = GlobalSumPooling(data_format='channels_last')(currentOutput)

    currentOutput = tf.keras.layers.Concatenate()(
        [size_value, n_patches_hot_encoded_value])
    for i in range(architecture_dict['n_layers']):
        dense_output = tf.keras.layers.Dense(architecture_dict['m_b'], activation='linear')(currentOutput)
        batchNorm = tf.keras.layers.BatchNormalization(momentum=0.75)(dense_output)
        activation = tf.keras.layers.ReLU()(batchNorm)
        currentOutput = activation
    size_and_n_patches_output = currentOutput

    concatenated_output = tf.keras.layers.Concatenate()(
        [global_pooling_output, size_and_n_patches_output])

    currentOutput = concatenated_output
    for i in range(architecture_dict['n_layers']):
        dense_output = tf.keras.layers.Dense(architecture_dict['m_c'], activation='linear')(currentOutput)
        batchNorm = tf.keras.layers.BatchNormalization(momentum=0.75)(dense_output)
        activation = tf.keras.layers.ReLU()(batchNorm)
        currentOutput = activation

    before_sigmoid_output = currentOutput

    output = tf.keras.layers.Dense(1, activation='sigmoid')(before_sigmoid_output)
    model = tf.keras.Model(inputs=[input_data, size_value, n_patches_hot_encoded_value], outputs=output)
    return model


def k_computation(prediction, training_ub_ratio,with_class_weights = False):
    if with_class_weights:
        training_ub_ratio = 0.5
    val = 1 - prediction
    if val == 0:
        return
    K = ((1 - training_ub_ratio) * prediction) / ((training_ub_ratio) * (val))
    return K


def prediction_function_using_bayes_factor_computation(priorUb, KValue):
    finalPrediction = float((KValue * priorUb) / ((KValue * priorUb) + (1 - priorUb)))
    return finalPrediction


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


def list_to_tsv(list_of_strings, output_file):
    with open(output_file, 'w') as f:
        for string in list_of_strings:
            f.write(string + '\t')


def createYhatGroupsFromPredictions(predictions, dictsForTraining, testOn='cv'):
    yhat_groups = []
    cnt = 0
    for i in range(len(dictsForTraining)):
        y = 'y_' + testOn
        yhat_groups.append([predictions[i] for i in range(cnt, cnt + dictsForTraining[i][y].size)])
        cnt += dictsForTraining[i][y].size
    return yhat_groups


def createInfoCsv(yhat_groups, dictsForTraining, allInfoDicts, dataDictPath, outputPath, testOn='cv'):
    data_dict = loadPickle(dataDictPath)
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


def getLabelsPredictionsAndArchitectureOfBestArchitecture(gridSearchDir):
    totalAucs = loadPickle(os.path.join(gridSearchDir, 'totalAucs.pkl'))
    totalAucs.sort(key=lambda x: -x[1])
    bestArchitecture = totalAucs[0][0]
    m_a = bestArchitecture[0]
    m_b = bestArchitecture[1]
    m_c = bestArchitecture[2]
    layers = bestArchitecture[3]
    predictionsAndLabels = loadPickle(
        os.path.join(gridSearchDir, 'predictions_labels_' + str(layers) + ' ' + str(m_a) + '.pkl'))
    for i in range(len(predictionsAndLabels)):
        if predictionsAndLabels[i][0][1] == m_b and predictionsAndLabels[i][0][2] == m_c:
            predictions = predictionsAndLabels[i][1]
            predictions = np.array([val[0] for val in predictions])
            labels = predictionsAndLabels[i][2]
            break
    return predictions, labels, bestArchitecture

def save_grid_search_results(grid_results,results_folder):
    #save the grid search results to a CSV file
    results_df = pd.DataFrame(grid_results)
    #sort the results by pr_auc
    results_df = results_df.sort_values(by='val_metric', ascending=False)
    results_df.to_csv(os.path.join(results_folder, 'grid_search_results.csv'), index=False)

def save_architecture_test_results(architecture_test_predictions,architecture_test_labels,results_architecture_folder):
    for i in range(len(architecture_test_predictions)):
        fold_results_folder_path = os.path.join(results_architecture_folder,f'fold_{i}')
        os.makedirs(fold_results_folder_path,exist_ok=True)
        np.save(os.path.join(fold_results_folder_path,'predictions.npy'),architecture_test_predictions[i])
        np.save(os.path.join(fold_results_folder_path,'labels.npy'),architecture_test_labels[i])
        header = f'precition_recall_curve_{i}'
        save_path = os.path.join(fold_results_folder_path, f'{header}.png')
        plot_precision_recall(architecture_test_predictions[i], architecture_test_labels[i],header,save_path)
    all_test_predictions = np.concatenate(architecture_test_predictions)
    all_test_labels = np.concatenate(architecture_test_labels)
    
    header = 'precition_recall_curve'
    save_path = os.path.join(results_architecture_folder, f'{header}.png')
    plot_precision_recall(all_test_predictions, all_test_labels,header,save_path)
    np.save(os.path.join(results_architecture_folder,'all_predictions.npy'),all_test_predictions)
    np.save(os.path.join(results_architecture_folder,'all_labels.npy'),all_test_labels)


# def createCSVFileFromResults(gridSearchDir, trainingDictsDir, dirName):
#     predictions, labels, bestArchitecture = getLabelsPredictionsAndArchitectureOfBestArchitecture(gridSearchDir)
#     allInfoDicts = loadPickle(os.path.join(trainingDictsDir, 'allInfoDicts.pkl'))
#     dictsForTraining = loadPickle(os.path.join(trainingDictsDir, 'dictsForTraining.pkl'))
#     dataDictPath = os.path.join(os.path.join(path.GoPath, 'idmapping_2023_12_26.tsv'), 'AllOrganizemsDataDict.pkl')
#     yhat_groups = createYhatGroupsFromPredictions(predictions, dictsForTraining)
#     outputPath = os.path.join(gridSearchDir, 'results_' + dirName + '.csv')
#     print(outputPath)
#     createInfoCsv(yhat_groups, dictsForTraining, allInfoDicts, dataDictPath, outputPath)
