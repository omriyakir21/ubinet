import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import numpy as np
import paths
from data_preparation.ScanNet.db_creation_scanNet_utils import save_as_pickle,load_as_pickle, THREE_LETTERS_TO_SINGLE_AA_DICT, aa_out_of_chain
from data_preparation.patch_to_score.data_development_utils import Protein, MAX_NUMBER_OF_COMPONENTS, transform_protein_data_list
from models.patch_to_score.patch_to_score_MLP_utils import k_computation, prediction_function_using_bayes_factor_computation
from results.patch_to_score.patch_to_score_result_analysis import get_best_architecture_path
import tensorflow as tf
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import auc
from typing import Dict, List
import pdb

def str_patch_from_list_indexes(residues, list_locations):
    myList = [THREE_LETTERS_TO_SINGLE_AA_DICT[residues[index].resname] + str(residues[index].id[1]) for index in
              list_locations]
    strPatch = ','.join(myList)
    return strPatch


def find_model_number(uniprot,uniprots_sets):
    for i in range(5):
        if uniprot in uniprots_sets[i]:
            return i
    return None


# def change_problamatic_values():
#     dataDictPath = os.path.join(os.path.join(path.GoPath, 'idmapping_2023_12_26.tsv'), 'AllOrganizemsDataDict.pkl')
#     data_dict = utils.loadPickle(dataDictPath)
#     entrysWithDifferentKeyNames = [(key, data_dict[key]['Entry']) for key in data_dict.keys() if
#                                    key != data_dict[key]['Entry']]
#     differentNamesDict = {}
#     for i in range(len(entrysWithDifferentKeyNames)):
#         differentNamesDict[entrysWithDifferentKeyNames[i][1]] = entrysWithDifferentKeyNames[i][0]

#     df = pd.read_csv(os.path.join(modelsDir, 'results_final_modelwith_evolution_50_plddt_all_organizems_15_4.csv'))

#     # Replace values in the 'Entry' column
#     df['Entry'].replace(differentNamesDict, inplace=True)

#     # Save the modified DataFrame back to a CSV file
#     df.to_csv(os.path.join(modelsDir, 'results_final_modelwith_evolution_50_plddt_all_organizems_fixed_15_4.csv'),
#               index=False)

#     print("Replacement complete.")


# def sort_locations(componentsLocations, sorted_indices):
#     sortedLocations = []
#     for i in range(len(componentsLocations)):
#         sortedLocations.append(componentsLocations[sorted_indices[i]])
#     return sortedLocations


# def sort_best_patches_from_uniprot(uniprot):
#     if uniprot not in aggragate.allPredictions['dict_resids'].keys():
#         raise Exception("uniprot " + str(uniprot) + " not in the DB")
#     modelIndex = find_model_number(uniprot)
#     model = models[modelIndex]
#     trainingUbRatio = np.mean(allInfoDicts[modelIndex]['y_train'])
#     protein = aggragate.Protein(uniprot, plddtThreshold)
#     tuples = protein.connectedComponentsTuples
#     components = np.array([[tup[0], tup[1], tup[2], tup[3]] for tup in tuples])
#     componentsLocations = [tup[4] for tup in tuples]
#     n_patches = 0
#     if len(tuples) == 0:
#         return [None for i in range(10)], [None for i in range(10)]
#     # SORT BY UB BINDING PROB
#     sorted_indices = tf.argsort(components[:, 1])
#     sorted_tensor = tf.gather(components, sorted_indices)
#     sortedTensorListed = [sorted_tensor]
#     utils.Scale4DUtil(sortedTensorListed, sizeComponentScaler, averageUbBindingScaler, plddtScaler)
#     sortedScaledPadded = tf.keras.preprocessing.sequence.pad_sequences(
#         sortedTensorListed, padding="post", maxlen=maxNumberOfPatches, dtype='float32')
#     n_patches = np.array([np.min([maxNumberOfPatches, sorted_tensor.shape[0]])])
#     n_patches_encoded = utils.hotOneEncodeNPatches(n_patches)
#     sortedLocations = sort_locations(componentsLocations, sorted_indices)
#     sortedLocationsCutted = sortedLocations[:n_patches[0]]

#     size = protein.size
#     sizeScaled = proteinSizeScaler.transform(np.array([size]).reshape(-1, 1))

#     yhat = model.predict([sortedScaledPadded, sizeScaled, n_patches_encoded])
#     KValue = utils.KComputation(yhat[0], trainingUbRatio)
#     inferencePrediction = utils.predictionFunctionUsingBayesFactorComputation(0.05, KValue)

#     significance = [None for _ in range(n_patches[0])]
#     for i in range(n_patches[0]):
#         newComponents = np.delete(sortedScaledPadded, i, axis=1)
#         location = sortedLocationsCutted[i]
#         new_n_patches_encoded = utils.hotOneEncodeNPatches(n_patches - 1)
#         newYhat = model.predict([newComponents, sizeScaled, new_n_patches_encoded])
#         newKValue = utils.KComputation(newYhat[0], trainingUbRatio)
#         newInferencePrediction = utils.predictionFunctionUsingBayesFactorComputation(0.05, newKValue)
#         significance[i] = (i, inferencePrediction - newInferencePrediction)

#     significance.sort(key=lambda x: -x[1])

#     strPatches, significance10 = create_str_patches_and_Significance10(significance, sortedLocationsCutted, uniprot,
#                                                                    protein)
#     return strPatches, significance10

# def createCsvForType(type, numOfType):
#     finalReslutsPath = os.path.join(modelsDir,
#                                     'results_final_modelwith_evolution_50_plddt_all_organizems_fixed_15_4.csv')
#     df = pd.read_csv(finalReslutsPath)
#     typeDf = df[df['type'] == type]
#     sortedDf = typeDf.sort_values(by='Inference Prediction 0.05 prior', ascending=False)
#     sortedCutted = sortedDf.head(numOfType)
#     uniprots = sortedCutted['Entry'].to_list()
#     strPatchesLists = [[] for i in range(10)]
#     significanceLists = [[] for i in range(10)]
#     for i in range(numOfType):
#         print(uniprots[i])
#         strPatches, significance10 = sortBestPatchesFromUniprot(uniprots[i])
#         for j in range(10):
#             strPatchesLists[j].append(strPatches[j])
#             significanceLists[j].append(significance10[j])

#     for i in range(10):
#         sortedCutted['Patch' + str(i)] = strPatchesLists[i]
#         sortedCutted['Reduced Probability' + str(i)] = significanceLists[i]
#     sortedCutted.to_csv(os.path.join(modelsDir, type + '.csv'), index=False)
#     return sortedCutted

def predict_for_protein(uniprot,scaled_size, scaled_components, encoded_components,models):
    modelIndex = find_model_number(uniprot)
    if modelIndex is None:
        yhat = np.mean([model.predict([scaled_components, scaled_size, encoded_components]) for model in models], axis=0)
    else:
        model = models[modelIndex]
        # pdb.set_trace()
        yhat = model.predict([scaled_components, scaled_size, encoded_components])
    return yhat

def get_patch_to_score_prediction_for_uniprots(all_predictions,plddt_threshold):
    uniprots = all_predictions['dict_resids'].keys()
    proteins = [Protein(uniprot, plddt_threshold, all_predictions) for uniprot in uniprots]
    scaled_sizes, scaled_components_list, encoded_components_list = (
    transform_protein_data_list(proteins,
            os.path.join(paths.scalers_path, 'scaler_size.pkl'),
            os.path.join(paths.scalers_path, 'scaler_components.pkl'),
            os.path.join(paths.scalers_path, 'encoder.pkl'),
            MAX_NUMBER_OF_COMPONENTS))
    dir_name = os.path.join(paths.with_MSA_patch_to_score_dir, "50_plddt")
    best_architecture_path = get_best_architecture_path(dir_name)
    predictions = []
    models = []
    models_dir_path = get_best_architecture_path(dir_name)
    for filename in os.listdir(models_dir_path):
        if filename.endswith('.keras'):
            models.append(tf.keras.models.load_model(os.path.join(models_dir_path, filename)))
    for i in range(len(proteins)):
        yhat = predict_for_protein(uniprots[i],scaled_sizes[i,:], scaled_components_list[i,:,:], encoded_components_list[i,:],models) 
        predictions.append(yhat)
    return predictions

def create_training_ub_ratio(all_predictions):
    all_predictions_ubiq = all_predictions['dict_predictions_ubiquitin']
    all_ubiq_predictions = []
    for ubiq_predictions in all_predictions_ubiq.values():
        all_ubiq_predictions.extend(list(ubiq_predictions))
    ub_ratio = np.mean(np.array(all_ubiq_predictions))
    print(f'ub_ratio: {ub_ratio}')
    save_as_pickle(all_ubiq_predictions, os.path.join(paths.patch_to_score_data_for_training_path, 'ub_ratio.pkl'))

def get_top_patches_indices(protein):
    components = protein.connectedComponentsTuples
    if len(components) == 0:
        return None
    sorted_components = sorted(components, key=lambda x: x[1], reverse=True)
    top_component_sets = [component[4] for component in sorted_components[:min(MAX_NUMBER_OF_COMPONENTS, len(components))]]
    return top_component_sets

def create_str_patches_and_Significance_for_protein(prediction,uniprot,scaled_size, scaled_components, encoded_components,models,ub_Ratio,protein):
    k_prediction = k_computation(prediction, ub_Ratio)
    inferece_prediction = prediction_function_using_bayes_factor_computation(0.05, k_prediction)
    significances = []
    str_patches = []
    number_of_patches = tf.argmax(encoded_components)
    if number_of_patches == 0:
        str_patches = [None for i in range(10)]
        significances = ['' for i in range(10)]
        return str_patches, significances
    
    indices = tf.argmax(encoded_components, axis=-1) 
    indices_subtracted = tf.maximum(indices - 1, 0)
    one_hot_tensor = tf.one_hot(indices_subtracted, depth=encoded_components.shape[-1])
    for i in range(number_of_patches):
        without_patch = tf.concat([scaled_components[:i],scaled_components[i+1:]], axis=1)
        padded_tensor = tf.keras.preprocessing.sequence.pad_sequences(
        without_patch, padding="post", maxlen=MAX_NUMBER_OF_COMPONENTS, dtype='float32')
        yhat = predict_for_protein(uniprot,scaled_size, padded_tensor, one_hot_tensor,models)
        k_value = k_computation(yhat, ub_Ratio)
        inference_yhat = prediction_function_using_bayes_factor_computation(0.05, k_value)
        significances.append(inferece_prediction - inference_yhat)
    
    sorted_significance_indices = np.argsort(np.array(significances))
    sorted_significance = [significances[i] for i in sorted_significance_indices]
    sorted_significance_padded =  sorted_significance + [0] * (MAX_NUMBER_OF_COMPONENTS - len(sorted_significance))
    top_component_sets = get_top_patches_indices(protein)
    sorted_component_sets = [top_component_sets[i] for i in sorted_significance_indices]
    for i in range(len(sorted_component_sets)):
        str_patches.append(str_patch_from_list_indexes(protein.residues, sorted_component_sets[i]))

    return sorted_significance_padded, str_patches

def create_str_patches_and_Significances(all_predictions,plddt_threshold,ub_ratio):
    predictions = get_patch_to_score_prediction_for_uniprots(all_predictions,plddt_threshold)
    uniprots = all_predictions['dict_resids'].keys()
    proteins = [Protein(uniprot, plddt_threshold, all_predictions) for uniprot in uniprots]
    scaled_sizes, scaled_components_list, encoded_components_list = (
    transform_protein_data_list(proteins,
            os.path.join(paths.scalers_path, 'scaler_size.pkl'),
            os.path.join(paths.scalers_path, 'scaler_components.pkl'),
            os.path.join(paths.scalers_path, 'encoder.pkl'),
            MAX_NUMBER_OF_COMPONENTS))
    
    for i in range(len(proteins)):
        create_str_patches_and_Significance_for_protein(predictions[i],uniprots[i],scaled_sizes[i,:], 
        scaled_components_list[i,:,:], encoded_components_list[i,:],models,ub_ratio,proteins[i])

def create_str_patches_and_Significance10(significance, sorted_locations_cutted, protein):
    structure = protein.getStructure()
    model = structure.child_list[0]
    assert (len(model) == 1)
    for chain in model:
        residues = aa_out_of_chain(chain)
    strPatches = [None for _ in range(10)]
    significance10 = [None for _ in range(10)]
    for i in range(len(significance)):
        strPatch = str_patch_from_list_indexes(residues, sorted_locations_cutted[significance[i][0]])
        strPatches[i] = strPatch
        significance10[i] = significance[i][1]
    return strPatches, significance10

def filter_dicts_by_keys(data: Dict[str, Dict[str, any]], keys: List[str]) -> Dict[str, Dict[str, any]]:
    """
    Filter a dictionary of dictionaries to include only the specified keys in each inner dictionary.

    Args:
    data (Dict[str, Dict[str, any]]): Dictionary of dictionaries to filter.
    keys (List[str]): List of keys to retain in each inner dictionary.

    Returns:
    Dict[str, Dict[str, any]]: New dictionary of dictionaries containing only the specified keys.
    """
    return {
        outer_key: {inner_key: value for inner_key, value in inner_dict.items() if inner_key in keys}
        for outer_key, inner_dict in data.items()
    }



def AF2_work():
    allPredictions = utils.loadPickle(os.path.join(path.ScanNetPredictionsPath, 'all_predictions_0304_MSA_True.pkl'))
    selected_keys = utils.loadPickle(os.path.join(path.AF2_multimerDir, 'selected_keys_' + str(200) + '.pkl'))
    predictions = np.array([get_patch_to_score_prediction_for_uniprots(uniprot) for uniprot in selected_keys])
    dict_sources = allPredictions['dict_sources']
    labels = np.array([0 if dict_sources[key] in utils.NegativeSources else 1 for key in selected_keys])
    ubinetPredictionsDict = {'predictions': predictions, 'labels': labels}
    utils.saveAsPickle(ubinetPredictionsDict, os.path.join(path.AF2_multimerDir, 'ubinetLabelsPredictions200'))
    precision, recall, thresholds = utils.precision_recall_curve(labels, predictions)
    sorted_indices = np.argsort(recall)
    sorted_precision = precision[sorted_indices]
    sorted_recall = recall[sorted_indices]
    aucScore = auc(sorted_recall, sorted_precision)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'Precision Recall curve (auc = {aucScore:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('AF2 iptm based predictor')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(path.AF2_multimerDir, 'ubinetPredictor'))
    plt.close()

if __name__ == '__main__':
    # all_predictions = load_as_pickle(os.path.join(paths.ScanNet_results_path, 'all_predictions_0304_MSA_True.pkl'))
    # create_training_ub_ratio(all_predictions)
    ub_ratio = load_as_pickle(os.path.join(paths.patch_to_score_data_for_training_path, 'ub_ratio.pkl'))
    # uniprots = ['O95793','Q8IYS0','Q86VN1','Q9NQL2','P07900','P53061']
    # all_predictions_sub = filter_dicts_by_keys(all_predictions, uniprots)
    # save_as_pickle(all_predictions_sub, os.path.join(paths.patch_to_score_results_path, 'all_predictions_sub.pkl'))
    all_predictions_sub = load_as_pickle(os.path.join(paths.patch_to_score_results_path, 'all_predictions_sub.pkl'))
    create_str_patches_and_Significances(all_predictions_sub,50,ub_ratio)
