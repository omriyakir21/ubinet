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


def create_merged_protein_object_dict():
    merged_dict = {}
    for i in range(len(dev_utils.indexes) - 1):
        d = load_as_pickle(os.path.join(paths.patches_dicts_path, 'proteinObjectsWithEvoluion' + str(i)))
        for key, value in d.items():
            merged_dict[key] = value
    return merged_dict


def create_data_relevant_for_training(max_number_of_components):
    cnt = 0
    all_uniprots = []
    all_sequences = []
    all_protein_paths = []
    all_data_components_flattend = []
    all_data_components = []
    all_data_protein_size = []
    all_data_number_of_components = []

    for i in range(len(dev_utils.indexes) - 1):
        print(cnt)
        cnt += 1
        d = load_as_pickle(os.path.join(paths.patches_dicts_path, 'proteinObjectsWithEvoluion' + str(i)))
        proteins = [protein for _, protein in d.items()]
        sequences = [protein.get_sequence() for protein in proteins]
        uniprots = [key for key, _ in d.items()]
        protein_paths = [os.path.join(paths.patches_dicts_path, f'proteinObjectsWithEvoluion{str(i)}') for _ in
                         range(len(uniprots))]

        data_components_flattend, data_protein_size, data_number_of_components, data_components = dev_utils.extract_protein_data(
            proteins,
            max_number_of_components)
        all_uniprots.extend(uniprots)
        all_sequences.extend(sequences)
        all_protein_paths.extend(protein_paths)
        all_data_components_flattend.extend(data_components_flattend)
        all_data_components.extend(data_components)
        all_data_protein_size.extend(data_protein_size)
        all_data_number_of_components.extend(data_number_of_components)
    return all_uniprots, all_sequences, all_protein_paths, all_data_components_flattend, all_data_protein_size, all_data_number_of_components, all_data_components


def create_patches(all_predictions):
    i = int(sys.argv[1])
    PLDDT_THRESHOLD = 50
    dev_utils.create_patches_dict(i, paths.patches_dicts_path, PLDDT_THRESHOLD, all_predictions)


if __name__ == "__main__":
    # CREATE PROTEIN OBJECTS, I'M DOING IT IN BATCHES
    # all_predictions = dev_utils.all_predictions
    # create_patches(all_predictions)

    # AFTER CREATING PROTEIN OBJECTS MERGE THEM TO 1 DICT
    MAX_NUMBER_OF_COMPONENTS = 10
    all_uniprots, all_sequences, all_protein_paths, all_data_components_flattend, all_data_protein_size, all_data_number_of_components, all_data_components = create_data_relevant_for_training(
        MAX_NUMBER_OF_COMPONENTS)
    save_as_pickle(all_uniprots, os.path.join(paths.patch_to_score_data_for_training_path, 'all_uniprots.pkl'))
    save_as_pickle(all_sequences, os.path.join(paths.patch_to_score_data_for_training_path, 'all_sequences.pkl'))
    save_as_pickle(all_protein_paths,
                   os.path.join(paths.patch_to_score_data_for_training_path, 'all_protein_paths.pkl'))
    save_as_pickle(all_data_components_flattend,
                   os.path.join(paths.patch_to_score_data_for_training_path, 'all_data_components_flattend.pkl'))
    save_as_pickle(all_data_protein_size,
                   os.path.join(paths.patch_to_score_data_for_training_path, 'all_data_protein_size.pkl'))
    save_as_pickle(all_data_number_of_components,
                   os.path.join(paths.patch_to_score_data_for_training_path, 'all_data_number_of_components.pkl'))
    save_as_pickle(all_data_components,
                   os.path.join(paths.patch_to_score_data_for_training_path, 'all_data_components.pkl'))

    all_uniprots = load_as_pickle(os.path.join(paths.patch_to_score_data_for_training_path, 'all_uniprots.pkl'))
    all_sequences = load_as_pickle(os.path.join(paths.patch_to_score_data_for_training_path, 'all_sequences.pkl'))
    all_protein_paths = load_as_pickle(
        os.path.join(paths.patch_to_score_data_for_training_path, 'all_protein_paths.pkl'))
    all_data_components_flattend = load_as_pickle(
        os.path.join(paths.patch_to_score_data_for_training_path, 'all_data_components_flattend.pkl'))
    all_data_protein_size = load_as_pickle(
        os.path.join(paths.patch_to_score_data_for_training_path, 'all_data_protein_size.pkl'))
    all_data_number_of_components = load_as_pickle(
        os.path.join(paths.patch_to_score_data_for_training_path, 'all_data_number_of_components.pkl'))
    all_data_components = load_as_pickle(
        os.path.join(paths.patch_to_score_data_for_training_path, 'all_data_components.pkl'))

    # CREATE SCALERS
    # dev_utils.fit_protein_data(all_data_components_flattend, all_data_protein_size, all_data_number_of_components,
    #                            paths.scalers_path, MAX_NUMBER_OF_COMPONENTS)

    # CREATE SCALED DATA FOR TRAINING
    # scaled_sizes, scaled_components_list, encoded_components_list = (
    #     dev_utils.transform_protein_data_list(all_proteins,
    #                                           os.path.join(paths.scalers_path, 'scaler_size.pkl'),
    #                                           os.path.join(paths.scalers_path, 'scaler_components.pkl'),
    #                                           os.path.join(paths.scalers_path, 'encoder.pkl'),
    #                                           MAX_NUMBER_OF_COMPONENTS))
    #
    # save_as_pickle(scaled_sizes, os.path.join(paths.patch_to_score_data_for_training_path, 'scaled_sizes'))
    # save_as_pickle(scaled_components_list,
    #                os.path.join(paths.patch_to_score_data_for_training_path, 'scaled_components_list'))
    # save_as_pickle(encoded_components_list,
    #                os.path.join(paths.patch_to_score_data_for_training_path, 'encoded_components_list'))

    # scaled_sizes = load_as_pickle(os.path.join(paths.patch_to_score_data_for_training_path, 'scaled_sizes'))
    # scaled_components_list = load_as_pickle(
    #     os.path.join(paths.patch_to_score_data_for_training_path, 'scaled_components_list'))
    # encoded_components_list = load_as_pickle(
    #     os.path.join(paths.patch_to_score_data_for_training_path, 'encoded_components_list'))
    #
    # # PARTIOTION THE DATA BY SEQUENCE LIKELIHOOD
    # cluster_indices, representative_indices = cluster_sequences(all_sequences, seqid=0.5, coverage=0.4,
    #                                                             path2mmseqstmp=paths.tmp_path,
    #                                                             path2mmseqs=paths.mmseqs_exec_path)
    # save_as_pickle(cluster_indices, os.path.join(paths.patch_to_score_data_for_training_path, 'cluster_indices'))
    # load_as_pickle(cluster_indices, os.path.join(paths.patch_to_score_data_for_training_path, 'cluster_indices'))
    # clusters_participants_list = partition_utils.create_cluster_participants_indices(cluster_indices)
    # cluster_sizes = [l.size for l in clusters_participants_list]
    # cluster_sizes_and_indices = [(i, cluster_sizes[i]) for i in range(len(cluster_sizes))]
    # sublists, sublists_sum = partition_utils.divide_clusters(cluster_sizes_and_indices)
    # print(f'sublists :{sublists}')
    # print(f'sublist sums :{sublists_sum}')
    # groups_indices = [partition_utils.get_uniprot_indices_for_groups(cluster_indices, sublists, fold_num) for fold_num
    #                   in
    #                   range(5)]

    # CREATE TRAINING DICTS
    # folds_training_dicts = dev_utils.create_training_folds(groups_indices,
    #                                                        os.path.join(paths.patch_to_score_data_for_training_path,
    #                                                                     'scaled_sizes'),
    #                                                        os.path.join(paths.patch_to_score_data_for_training_path,
    #                                                                     'scaled_components_list'),
    #                                                        os.path.join(paths.patch_to_score_data_for_training_path,
    #                                                                     'encoded_components_list'),
    #                                                        all_uniprots)

    # CREATE DATA FOR TRAINING

    # trainingDataDir = os.path.join(path.predictionsToDataSetDir, dirName)
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
