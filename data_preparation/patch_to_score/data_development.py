import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import paths
from data_preparation.patch_to_score import data_development_utils as development_utils
from data_preparation.ScanNet.db_creation_scanNet_utils import save_as_pickle, load_as_pickle
import data_development_utils as dev_utils
import numpy as np
import protein_level_data_partition_utils as partition_utils
from data_preparation.ScanNet.create_tables_and_weights import cluster_sequences
import pickle
import pdb
import tensorflow as tf

def save_data_for_training(uniprots,sources,protein_paths):
    scaled_sizes, scaled_components_list, encoded_components_list = (
    dev_utils.transform_protein_data_list(proteins,
                                              os.path.join(paths.scalers_path, 'scaler_size.pkl'),
                                              os.path.join(paths.scalers_path, 'scaler_components.pkl'),
                                              os.path.join(paths.scalers_path, 'encoder.pkl'),
                                              MAX_NUMBER_OF_COMPONENTS))
    labels = tf.convert_to_tensor([0 if source in dev_utils.NEGATIVE_SOURCES else 1 for source in sources])
    dev_utils.save_as_tensor(scaled_sizes, os.path.join(paths.patch_to_score_data_for_training_path, 'scaled_sizes.tf'))
    dev_utils.save_as_tensor(scaled_components_list, os.path.join(paths.patch_to_score_data_for_training_path, 'scaled_components_list.tf'))
    dev_utils.save_as_tensor(encoded_components_list, os.path.join(paths.patch_to_score_data_for_training_path, 'encoded_components_list.tf'))
    dev_utils.save_as_tensor(labels,os.path.join(paths.patch_to_score_data_for_training_path, 'labels.tf'))
    save_as_pickle(uniprots, os.path.join(paths.patch_to_score_data_for_training_path, 'uniprots.pkl'))
    save_as_pickle(sources,os.path.join(paths.patch_to_score_data_for_training_path, 'sources.pkl'))
    save_as_pickle(protein_paths,os.path.join(paths.patch_to_score_data_for_training_path, 'protein_paths.pkl'))
    


def create_merged_protein_object_dict():
    merged_dict = {}
    for i in range(len(dev_utils.indexes) - 1):
        d = load_as_pickle(os.path.join(paths.patches_dicts_path, 'proteinObjectsWithEvoluion' + str(i)))
        for key, value in d.items():
            merged_dict[key] = value
    return merged_dict


def create_data_relevant_for_training(max_number_of_components, merged_dict):
    proteins = [protein for _, protein in merged_dict.items()]
    sequences = [protein.get_sequence() for protein in proteins]
    sources = [protein.source for protein in proteins]
    uniprots = [key for key, _ in merged_dict.items()]
    protein_paths = [os.path.join(paths.patches_dicts_path, f'proteinObjectsWithEvoluion{str(i // 1500)}') for i in
                     range(len(uniprots))]

    data_components_flattend, data_protein_size, data_number_of_components, data_components = dev_utils.extract_protein_data(
        proteins,
        max_number_of_components)

    assert (len(uniprots) == len(sequences) == len(protein_paths) == len(
        data_components) == len(data_protein_size) == len(sources) == len(data_number_of_components))
    return uniprots, sequences, protein_paths, data_components_flattend, data_protein_size, data_number_of_components, data_components, sources


def create_patches(all_predictions):
    i = int(sys.argv[1])
    PLDDT_THRESHOLD = 50
    dev_utils.create_patches_dict(i, paths.patches_dicts_path, PLDDT_THRESHOLD, all_predictions)


def create_small_sample_dict(merge_dict):
    small_dict = {}
    for i, (key, value) in enumerate(merge_dict.items()):
        if i == 10:
            break
        small_dict[key] = value
    save_as_pickle(small_dict, os.path.join(paths.patches_dicts_path, 'small_sample_dict.pkl'))

def partition_to_folds_and_save(sequences):
    cluster_indices, representative_indices = cluster_sequences(sequences, seqid=0.5, coverage=0.4,
                                                                path2mmseqstmp=paths.tmp_path,
                                                                path2mmseqs=paths.mmseqs_exec_path)
    save_as_pickle(cluster_indices, os.path.join(paths.patch_to_score_data_for_training_path, 'cluster_indices.pkl'))
    clusters_participants_list = partition_utils.create_cluster_participants_indices(cluster_indices)
    cluster_sizes = [l.size for l in clusters_participants_list]
    cluster_sizes_and_indices = [(i, cluster_sizes[i]) for i in range(len(cluster_sizes))]
    sublists, sublists_sum = partition_utils.divide_clusters(cluster_sizes_and_indices)
    groups_indices = [partition_utils.get_uniprot_indices_for_groups(clusters_participants_list, sublists, fold_num) for fold_num
                      in
                      range(5)] 
    # for group_indices in groups_indices:
        # print(f'group_indices is {group_indices}')
        # print(f'max group indice is :{np.max(group_indices)}')
    save_as_pickle(groups_indices, os.path.join(paths.patch_to_score_data_for_training_path, 'groups_indices.pkl'))

    # print(f'cluster_indices: {cluster_indices}',flush=True)
    # print(f'cluster_participant-list: {clusters_participants_list}',flush=True)
    # print(f'cluster_sizes: {cluster_sizes}',flush=True)
    # print(f'sublists: {sublists}',flush = True)
    # print(f'sublists_sum: {sublists_sum}',flush = True)
    # print(f'group_indices: {groups_indices}')
  
    # CREATE TRAINING DICTS
    folds_training_dicts = dev_utils.create_training_folds(groups_indices,
                                                           os.path.join(paths.patch_to_score_data_for_training_path,
                                                                        'scaled_sizes.tf'),
                                                           os.path.join(paths.patch_to_score_data_for_training_path,
                                                                        'scaled_components_list.tf'),
                                                           os.path.join(paths.patch_to_score_data_for_training_path,
                                                                        'encoded_components_list.tf'),
                                                           os.path.join(paths.patch_to_score_data_for_training_path, 'uniprots.pkl'),
                                                           os.path.join(paths.patch_to_score_data_for_training_path, 'labels.tf'))
    print(f'before saving folds dict')
    save_as_pickle(folds_training_dicts,os.path.join(paths.patch_to_score_data_for_training_path,
                                                                        'folds_traning_dicts.pkl'))


if __name__ == "__main__":
    # CREATE PROTEIN OBJECTS, I'M DOING IT IN BATCHES
    # all_predictions = dev_utils.all_predictions
    # create_patches(all_predictions)

    MAX_NUMBER_OF_COMPONENTS = 10
    # merged_dict = create_merged_protein_object_dict()
    # save_as_pickle(merged_dict, os.path.join(paths.patches_dicts_path, 'merged_protein_objects_with_evolution')
    merged_dict = load_as_pickle(os.path.join(paths.patches_dicts_path, 'merged_protein_objects_with_evolution'))
    # merged_dict = load_as_pickle(os.path.join(paths.patches_dicts_path, 'proteinObjectsWithEvoluion0'))
    # create_small_sample_dict(merged_dict)
    # merged_dict = load_as_pickle(os.path.join(paths.patches_dicts_path, 'small_sample_dict.pkl'))
    uniprots, sequences, protein_paths, data_components_flattend, data_protein_size, data_number_of_components, data_components, sources = create_data_relevant_for_training(
        MAX_NUMBER_OF_COMPONENTS, merged_dict)
    proteins = [protein for _, protein in merged_dict.items()]
    # CREATE SCALERS
   
    dev_utils.fit_protein_data(np.array(data_components_flattend), np.array(data_protein_size),  np.array(data_number_of_components),
                               paths.scalers_path, MAX_NUMBER_OF_COMPONENTS)

    # CREATE SCALED DATA FOR TRAINING
    save_data_for_training(uniprots,sources,protein_paths)

    # PARTIOTION THE DATA BY SEQUENCE LIKELIHOOD
    partition_to_folds_and_save(sequences)
 
    
    # gridSearchDir = os.path.join(path.aggregateFunctionMLPDir, 'MLP_MSA_val_AUC_stoppage_' + dirName)

    # #
    # trainingDictsDir = os.path.join(trainingDataDir, 'trainingDicts')

    # plotPlddtHistogramForPositivieAndProteome(allPredictions)

    # FROM HERE FOLLOWS IN ONE RUN
    # PKL ALL THE COMPONENTS TOGETHER AND CREATE LABELS FROM THE PATCHES LIST
    # components = pklComponentsOutOfProteinObjects(trainingDataDir)
    # labels = pklLabels(components, trainingDataDir)

    # CREATE DATA FOR TRAINING (allInfoDicts and dictForTraining)
    # componentsDir = os.path.join(trainingDataDir, 'components')
    # componentsPath = os.path.join(componentsDir, 'components.pkl')
    # labelsDir = os.path.join(trainingDataDir, 'labels')
    # labelsPath = os.path.join(labelsDir, 'labels.pkl')
    # try:
    #     os.mkdir(trainingDictsDir)
    # except Exception as e:
    #     print(e)
    # allInfoDict, dictForTraining = utils.createDataForTraining(componentsPath, labelsPath, trainingDictsDir)

    # PARTITION THE DATA
    # proteinLevelDataPartition.create_x_y_groups('all_predictions_0304_MSA_True.pkl', trainingDataDir)

    # CREATE TRAIN TEST VALIDATION FOR ALL GROUPS
    # x_groups = loadPickle(os.path.join(trainingDictsDir, 'x_groups.pkl'))
    # y_groups = loadPickle(os.path.join(trainingDictsDir, 'y_groups.pkl'))

    # componentsGroups = loadPickle(os.path.join(trainingDictsDir, 'componentsGroups.pkl'))
    # sizesGroups = loadPickle(os.path.join(trainingDictsDir, 'sizesGroups.pkl'))
    # n_patchesGroups = loadPickle(os.path.join(trainingDictsDir, 'n_patchesGroups.pkl'))
    # allInfoDicts, dictsForTraining = utils.createTrainValidationTestForAllGroups(x_groups, y_groups, componentsGroups,
    #                                                                              sizesGroups, n_patchesGroups,
    #                                                                              trainingDictsDir)

    # CREATING THE CSV FILE
    # utils.createCSVFileFromResults(gridSearchDir, trainingDictsDir, dirName)

    # PLOT SUMMARY  FILES
    # createPRPlotFromResults(gridSearchDir)
    # createLogBayesDistributionPlotFromResults(gridSearchDir)
    # THATS IT FROM HERE IT IS NOT RELEVANT

    # CREATE COMBINED CSV
    # dirName2 = sys.argv[4]
    # plddtThreshold2 = int(sys.argv[5])
    # trainingDataDir2 = os.path.join(path.predictionsToDataSetDir, dirName2)
    # gridSearchDir2 = os.path.join(path.aggregateFunctionMLPDir, 'MLP_MSA_val_AUC_stoppage_' + dirName2)
    # # createCombinedCsv(gridSearchDir, dirName, gridSearchDir2, dirName2, plddtThreshold, plddtThreshold2)
    # createCombinedCsv(os.path.join(gridSearchDir,'finalModel'), dirName, os.path.join(gridSearchDir2,'finalModel'), dirName2, plddtThreshold, plddtThreshold2)

    # PLOT DUMMY BASELINE FOR AGGREGATE SCORING FUNCTION
    # plotDummyPRAUC(allPredictions)

    # !!!!
