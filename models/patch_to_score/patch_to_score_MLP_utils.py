# for array computations and loading data
import math
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import csv

from PIL.ImageOps import pad
from sklearn.utils import compute_class_weight

import paths
import numpy as np
from matplotlib import pyplot as plt

# for building linear regression models and preparing data
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, classification_report, roc_auc_score, roc_curve, precision_recall_curve, \
    auc
import pickle
# for building and training neural networks
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling1D, Reshape, Masking

maxNumberOfPatches = 10
from tensorflow.python.keras import backend
from data_preparation.patch_to_score.data_development_utils import NEGATIVE_SOURCES,POSITIVE_SOURCES


class GlobalSumPooling(GlobalAveragePooling1D):
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


def hotOneEncodeNPatches(n_patches_array):
    encoded = np.zeros((n_patches_array.shape[0], maxNumberOfPatches + 1))
    encoded[np.arange(n_patches_array.size), np.minimum(n_patches_array,
                                                        np.full(n_patches_array.shape, maxNumberOfPatches))] = 1
    return encoded


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


def plotPrecisionRecall(y_probs, labels, header):
    precision, recall, thresholds = precision_recall_curve(labels, y_probs)
    pr_auc = auc(recall, precision)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label='Precision-Recall Curve, AUC =' + str(pr_auc))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve ' + header)
    plt.legend()
    plt.grid(True)
    plt.show()


def createDataForTraining(componentsPath, labelsPath, outputDirPath):
    modelsDir = os.path.join(path.gridSearchDir, 'finalmodel')
    labels = loadPickle(labelsPath)
    tuplesLen4 = loadPickle(componentsPath)
    x_train, x_cv, x_test, y_train, y_cv, y_test = divideTrainValidationTest(tuplesLen4, labels)
    x_train_components, x_train_sizes, x_train_n_patches = divideXData(x_train)
    x_cv_components, x_cv_sizes, x_cv_n_patches = divideXData(x_cv)
    x_test_components, x_test_sizes, x_test_n_patches = divideXData(x_test)
    x_train_n_patches_encoded = hotOneEncodeNPatches(x_train_n_patches)
    x_cv_n_patches_encoded = hotOneEncodeNPatches(x_cv_n_patches)
    x_test_n_patches_encoded = hotOneEncodeNPatches(x_test_n_patches)
    sortPatches(x_train_components)
    sortPatches(x_cv_components)
    sortPatches(x_test_components)
    scaleXComponents4D(x_train_components, x_cv_components, x_test_components, modelsDir)
    x_train_sizes_scaled, x_cv_sizes_scaled, x_test_sizes_scaled = getScaleXSizes3D(x_train_sizes, x_cv_sizes,
                                                                                    x_test_sizes, modelsDir)
    x_train_components_scaled_padded, x_cv_components_scaled_padded, x_test_components_scaled_padded = padXValues(
        x_train_components, x_cv_components, x_test_components, maxNumberOfPatches)
    allInfoDict = {'x_train': x_train, 'x_cv': x_cv, 'x_test': x_test, 'y_train': y_train, 'y_cv': y_cv,
                   'y_test': y_test}
    dictForTraining = {'x_train_components_scaled_padded': x_train_components_scaled_padded,
                       'x_cv_components_scaled_padded': x_cv_components_scaled_padded,
                       'x_test_components_scaled_padded': x_test_components_scaled_padded,
                       'x_train_sizes_scaled': x_train_sizes_scaled, 'x_cv_sizes_scaled': x_cv_sizes_scaled,
                       'x_test_sizes_scaled': x_test_sizes_scaled,
                       'x_train_n_patches_encoded': x_train_n_patches_encoded,
                       'x_cv_n_patches_encoded': x_cv_n_patches_encoded,
                       'x_test_n_patches_encoded': x_test_n_patches_encoded,
                       'y_train': y_train, 'y_cv': y_cv,
                       'y_test': y_test
                       }

    saveAsPickle(allInfoDict, os.path.join(outputDirPath, 'allInfoDict'))
    saveAsPickle(dictForTraining, os.path.join(outputDirPath, 'dictForTraining'))
    return allInfoDict, dictForTraining


def saveScalersForFinalModel(finalModelPath, allInfoDictPath):
    allInfoDict = loadPickle(allInfoDictPath)
    x_train = allInfoDict['x_train']
    x_train_components, x_train_sizes, x_train_n_patches = divideXData(x_train)
    sortPatches(x_train_components)
    # CREATE SCALERS FOR COMPONENTS
    allTupleSizesTrain = np.concatenate([tuples[:, 0] for tuples in x_train_components if tuples.shape != (0,)])
    allTupleUbAveragesTrain = np.concatenate([tuples[:, 1] for tuples in x_train_components if tuples.shape != (0,)])
    allTuplePlddtTrain = np.concatenate([tuples[:, 3] for tuples in x_train_components if tuples.shape != (0,)])
    scalerSizeComponent = StandardScaler()
    scalerAverageUbBinding = StandardScaler()
    plddtScaler = StandardScaler()
    scalerSizeComponent.fit(allTupleSizesTrain.reshape((-1, 1)))
    scalerAverageUbBinding.fit(allTupleUbAveragesTrain.reshape((-1, 1)))
    plddtScaler.fit(allTuplePlddtTrain.reshape((-1, 1)))
    # CREATE SCALER FOR SIZE
    scalerSizeProtein = StandardScaler()
    scalerSizeProtein.fit(x_train_sizes.reshape((-1, 1)))
    saveAsPickle(scalerSizeComponent, os.path.join(finalModelPath, 'sizeComponentScaler'))
    saveAsPickle(scalerAverageUbBinding, os.path.join(finalModelPath, 'averageUbBindingScaler'))
    saveAsPickle(plddtScaler, os.path.join(finalModelPath, 'plddtScaler'))
    saveAsPickle(scalerSizeProtein, os.path.join(finalModelPath, 'proteinSizeScaler'))


def simpleModelTraining():
    model = Sequential(
        [
            tf.keras.Input(shape=(maxNumberOfPatches, 3)),
            Masking(mask_value=0.0),
            Dense(36, activation='relu'),
            Dense(25, activation='relu'),
            Dense(16, activation='relu'),
            GlobalSumPooling(data_format='channels_last'),
            Dense(1, activation='sigmoid')
        ],
        name='model_2'
    )
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=['accuracy']
    )
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    # Convert class weights to a dictionary
    class_weight = {i: class_weights[i] for i in range(len(class_weights))}

    model.fit(
        x_train_components_scaled_padded, y_train,
        epochs=300,
        verbose=1,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=6)],
        batch_size=1024,
        class_weight=class_weight

    )

    yhat_train = model.predict(x_train_components_scaled_padded)
    predictions_train = np.where(yhat_train >= 0.5, 1, 0)
    print(classification_report(y_train, predictions_train))


def createTrainValidationTestForAllGroups(x_groups, y_groups, componentsGroups, sizesGroups, n_patchesGroups,
                                          outputDir):
    dictsForTraining = []
    allInfoDicts = []
    for i in range(len(x_groups)):
        x_cv_components_scaled_padded = componentsGroups[i % 5]
        x_test_components_scaled_padded = componentsGroups[(i + 1) % 5]
        x_cv_sizes_scaled = sizesGroups[i % 5]
        x_test_sizes_scaled = sizesGroups[(i + 1) % 5]
        x_cv_n_patches_encoded = n_patchesGroups[i % 5]
        x_test_n_patches_encoded = n_patchesGroups[(i + 1) % 5]
        x_cv = x_groups[i % 5]
        x_test = x_groups[(i + 1) % 5]
        y_cv = y_groups[i % 5]
        y_test = y_groups[(i + 1) % 5]
        groupsForComponentsXTrain = [componentsGroups[j] for j in range(len(componentsGroups)) if
                                     j != i % 5 and j != (i + 1) % 5]
        x_train_components_scaled_padded = np.concatenate(
            (groupsForComponentsXTrain[0], groupsForComponentsXTrain[1], groupsForComponentsXTrain[2]), axis=0)
        groupsForSizesXTrain = [sizesGroups[j] for j in range(len(sizesGroups)) if j != i % 5 and j != (i + 1) % 5]
        x_train_sizes_scaled = np.concatenate(
            (groupsForSizesXTrain[0], groupsForSizesXTrain[1], groupsForSizesXTrain[2]), axis=0)
        groupsForNPatchesXTrain = [n_patchesGroups[j] for j in range(len(n_patchesGroups)) if
                                   j != i % 5 and j != (i + 1) % 5]
        x_train_n_patches_encoded = np.concatenate(
            (groupsForNPatchesXTrain[0], groupsForNPatchesXTrain[1], groupsForNPatchesXTrain[2]), axis=0)
        groupsForXTrain = [x_groups[j] for j in range(len(x_groups)) if j != i % 5 and j != (i + 1) % 5]
        groupsForYTrain = [y_groups[j] for j in range(len(y_groups)) if j != i % 5 and j != (i + 1) % 5]
        x_train = []
        y_train = []
        for i in range(len(groupsForXTrain)):
            x_train.extend(groupsForXTrain[i])
            y_train.extend(groupsForYTrain[i])

        # x_train_components, x_train_sizes, x_train_n_patches = divideXData(x_train)
        # x_cv_components, x_cv_sizes, x_cv_n_patches = divideXData(x_cv)
        # x_test_components, x_test_sizes, x_test_n_patches = divideXData(x_test)
        # x_train_n_patches_encoded = hotOneEncodeNPatches(x_train_n_patches)
        # x_cv_n_patches_encoded = hotOneEncodeNPatches(x_cv_n_patches)
        # x_test_n_patches_encoded = hotOneEncodeNPatches(x_test_n_patches)
        # sortPatches(x_train_components)
        # sortPatches(x_cv_components)
        # sortPatches(x_test_components)
        # scaleXComponents4D(x_train_components, x_cv_components, x_test_components)
        # x_train_sizes_scaled, x_cv_sizes_scaled, x_test_sizes_scaled = getScaleXSizes3D(x_train_sizes, x_cv_sizes,
        #                                                                                 x_test_sizes)
        # x_train_components_scaled_padded, x_cv_components_scaled_padded, x_test_components_scaled_padded = padXValues(
        #     x_train_components, x_cv_components, x_test_components, maxNumberOfPatches)
        allInfoDict = {'x_train': x_train, 'x_cv': x_cv, 'x_test': x_test, 'y_train': y_train, 'y_cv': y_cv,
                       'y_test': y_test}
        dictForTraining = {'x_train_components_scaled_padded': x_train_components_scaled_padded,
                           'x_cv_components_scaled_padded': x_cv_components_scaled_padded,
                           'x_test_components_scaled_padded': x_test_components_scaled_padded,
                           'x_train_sizes_scaled': x_train_sizes_scaled, 'x_cv_sizes_scaled': x_cv_sizes_scaled,
                           'x_test_sizes_scaled': x_test_sizes_scaled,
                           'x_train_n_patches_encoded': x_train_n_patches_encoded,
                           'x_cv_n_patches_encoded': x_cv_n_patches_encoded,
                           'x_test_n_patches_encoded': x_test_n_patches_encoded,
                           'y_train': y_train, 'y_cv': y_cv,
                           'y_test': y_test
                           }

        dictsForTraining.append(dictForTraining)
        allInfoDicts.append(allInfoDict)
    saveAsPickle(allInfoDicts, os.path.join(outputDir, 'allInfoDicts'))
    saveAsPickle(dictsForTraining, os.path.join(outputDir, 'dictsForTraining'))
    return allInfoDicts, dictsForTraining


def buildModel(n_layers_component, n_layers_size, n_layers_n_patches, n_layers_final):
    # Define the input shape
    input_shape = (maxNumberOfPatches, 4)
    input_data = tf.keras.Input(shape=input_shape, name='sensor_input')
    size_value = tf.keras.Input(shape=(1,), name='extra_value_input')
    n_patches_hot_encoded_value = tf.keras.Input(shape=(maxNumberOfPatches + 1,), name='hot_encoded_value_input')
    masked_input = tf.keras.layers.Masking(mask_value=0.0)(input_data)

    currentOutput = masked_input
    for i in range(n_layers_component, 0, -1):
        dense_output = tf.keras.layers.Dense(pow(2, 3 + i), activation='linear')(currentOutput)
        batchNorm = tf.keras.layers.BatchNormalization()(dense_output)
        activation = tf.keras.layers.ReLU()(batchNorm)
        currentOutput = activation

    global_pooling_output = GlobalSumPooling(data_format='channels_last')(currentOutput)

    currentOutput = size_value
    for i in range(n_layers_size, 0, -1):
        dense_output = tf.keras.layers.Dense(pow(2, 3 + i), activation='linear')(currentOutput)
        batchNorm = tf.keras.layers.BatchNormalization()(dense_output)
        activation = tf.keras.layers.ReLU()(batchNorm)
        currentOutput = activation
    size_output = currentOutput

    currentOutput = n_patches_hot_encoded_value
    for i in range(n_layers_n_patches, 0, -1):
        dense_output = tf.keras.layers.Dense(pow(2, 3 + i), activation='linear')(currentOutput)
        batchNorm = tf.keras.layers.BatchNormalization()(dense_output)
        activation = tf.keras.layers.ReLU()(batchNorm)
        currentOutput = activation
    n_patches_output = currentOutput

    concatenated_output = tf.keras.layers.Concatenate()(
        [global_pooling_output, size_output, n_patches_output])

    currentOutput = concatenated_output
    for i in range(n_layers_final, 0, -1):
        dense_output = tf.keras.layers.Dense(pow(2, 3 + i), activation='linear')(currentOutput)
        batchNorm = tf.keras.layers.BatchNormalization()(dense_output)
        activation = tf.keras.layers.ReLU()(batchNorm)
        currentOutput = activation

    before_sigmoid_output = currentOutput

    output = tf.keras.layers.Dense(1, activation='sigmoid')(before_sigmoid_output)
    model = tf.keras.Model(inputs=[input_data, size_value, n_patches_hot_encoded_value], outputs=output)
    return model


def build_model_concat_size_and_n_patches_same_number_of_layers(m_a, m_b, m_c, n_layers):
    '''
    :param m_a: size of the hidden layers in the MLP of the components
    :param m_b: size of the hidden layers in the MLP of the concatenated size and number of patches
    :param m_c: size of the hidden layers in the MLP of the concatenated global sum output and size + n_patches MLP output
    :param n_layers: number of layers in each of the MLPs
    :return:
    '''
    # Define the input shape
    input_shape = (maxNumberOfPatches, 4)
    input_data = tf.keras.Input(shape=input_shape, name='sensor_input')
    size_value = tf.keras.Input(shape=(1,), name='extra_value_input')
    n_patches_hot_encoded_value = tf.keras.Input(shape=(maxNumberOfPatches + 1,), name='hot_encoded_value_input')
    masked_input = tf.keras.layers.Masking(mask_value=0.0)(input_data)

    currentOutput = masked_input
    for i in range(n_layers):
        dense_output = tf.keras.layers.Dense(m_a, activation='linear')(currentOutput)
        batchNorm = tf.keras.layers.BatchNormalization(momentum=0.75)(dense_output)
        activation = tf.keras.layers.ReLU()(batchNorm)
        currentOutput = activation

    global_pooling_output = GlobalSumPooling(data_format='channels_last')(currentOutput)

    currentOutput = tf.keras.layers.Concatenate()(
        [size_value, n_patches_hot_encoded_value])
    for i in range(n_layers):
        dense_output = tf.keras.layers.Dense(m_b, activation='linear')(currentOutput)
        batchNorm = tf.keras.layers.BatchNormalization(momentum=0.75)(dense_output)
        activation = tf.keras.layers.ReLU()(batchNorm)
        currentOutput = activation
    size_and_n_patches_output = currentOutput

    concatenated_output = tf.keras.layers.Concatenate()(
        [global_pooling_output, size_and_n_patches_output])

    currentOutput = concatenated_output
    for i in range(n_layers):
        dense_output = tf.keras.layers.Dense(m_c, activation='linear')(currentOutput)
        batchNorm = tf.keras.layers.BatchNormalization(momentum=0.75)(dense_output)
        activation = tf.keras.layers.ReLU()(batchNorm)
        currentOutput = activation

    before_sigmoid_output = currentOutput

    output = tf.keras.layers.Dense(1, activation='sigmoid')(before_sigmoid_output)
    model = tf.keras.Model(inputs=[input_data, size_value, n_patches_hot_encoded_value], outputs=output)
    return model


def KComputation(prediction, trainingUbRation):
    val = 1 - prediction
    if val == 0:
        return
    K = ((1 - trainingUbRation) * prediction) / ((trainingUbRation) * (val))
    return K


def predictionFunctionUsingBayesFactorComputation(priorUb, KValue):
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
        allKvalues.extend([KComputation(yhat_groups[i][j], trainingUbRatio) for j in range(len(yhat_groups[i]))])
    Inference5PercentPredictions = [predictionFunctionUsingBayesFactorComputation(0.05, KValue) for KValue in
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


def createCSVFileFromResults(gridSearchDir, trainingDictsDir, dirName):
    predictions, labels, bestArchitecture = getLabelsPredictionsAndArchitectureOfBestArchitecture(gridSearchDir)
    allInfoDicts = loadPickle(os.path.join(trainingDictsDir, 'allInfoDicts.pkl'))
    dictsForTraining = loadPickle(os.path.join(trainingDictsDir, 'dictsForTraining.pkl'))
    dataDictPath = os.path.join(os.path.join(path.GoPath, 'idmapping_2023_12_26.tsv'), 'AllOrganizemsDataDict.pkl')
    yhat_groups = createYhatGroupsFromPredictions(predictions, dictsForTraining)
    outputPath = os.path.join(gridSearchDir, 'results_' + dirName + '.csv')
    print(outputPath)
    createInfoCsv(yhat_groups, dictsForTraining, allInfoDicts, dataDictPath, outputPath)
