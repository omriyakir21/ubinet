import os
import csv
import pickle

import numpy as nps
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_curve, auc
import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras.layers import GlobalAveragePooling1D


@tf.keras.utils.register_keras_serializable(package="MyCustomLayersNoMlpB")
class GlobalSumPooling(GlobalAveragePooling1D):
    def __init__(self, data_format='channels_last', keepdims=False, **kwargs):
        super(GlobalSumPooling, self).__init__(
            data_format=data_format, **kwargs)
        self.keepdims = keepdims

    def call(self, inputs, mask=None):
        steps_axis = 1 if self.data_format == "channels_last" else 2
        if mask is not None:
            mask = tf.cast(mask, inputs[0].dtype)
            mask = tf.expand_dims(
                mask, 2 if self.data_format == "channels_last" else 1
            )
            inputs *= mask
            return backend.sum(
                inputs, axis=steps_axis, keepdims=self.keepdims
            )

    def get_config(self):
        config = super(GlobalSumPooling, self).get_config()
        config.update({
            "data_format": self.data_format,
            "keepdims": self.keepdims
        })
        return config


def saveAsPickle(object, fileName):
    with open(fileName + '.pkl', 'wb') as file:
        pickle.dump(object, file)


def loadPickle(fileName):
    with open(fileName, 'rb') as file:
        object = pickle.load(file)
        return object


def plot_precision_recall(y_probs, labels, header, save_path):
    precision, recall, _ = precision_recall_curve(labels, y_probs)
    pr_auc = auc(recall, precision)
    plt.figure(figsize=(8, 6))
    pr_auc = round(pr_auc, 3)
    plt.plot(recall, precision,
             label='Precision-Recall Curve, AUC =' + str(pr_auc))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(header)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def k_computation(prediction, training_ub_ratio, with_class_weights=False):
    if with_class_weights:
        training_ub_ratio = 0.5
    val = 1 - prediction
    if val == 0:
        return
    K = ((1 - training_ub_ratio) * prediction) / ((training_ub_ratio) * (val))
    return K


def prediction_function_using_bayes_factor_computation(priorUb, KValue):
    finalPrediction = float((KValue * priorUb) /
                            ((KValue * priorUb) + (1 - priorUb)))
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


def createInfoCsv(yhat_groups, dictsForTraining, allInfoDicts, dataDictPath, outputPath, testOn='cv'):
    data_dict = loadPickle(dataDictPath)
    allKvalues = []
    for i in range(len(dictsForTraining)):
        trainingUbRatio = np.mean(dictsForTraining[i]['y_train'])
        allKvalues.extend([k_computation(yhat_groups[i][j], trainingUbRatio)
                          for j in range(len(yhat_groups[i]))])
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
                uniDict = {
                    'Entry': allInfoDicts[i][x][j][1], 'Protein names': 'No Info', 'Organism': 'No Info'}
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
