import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import paths
from data_preparation.patch_to_score import data_development_utils as development_utils
from data_preparation.ScanNet.db_creation_scanNet_utils import save_as_pickle, load_as_pickle
import data_development_utils as dev_utils
import numpy as np


def create_patches():
    all_predictions = dev_utils.all_predictions
    i = sys.argv[1]
    j = sys.argv[2]
    PLDDT_THRESHOLD = 50
    for k in range(i, j + 1):
        dev_utils.create_patches_dict(k, paths.patches_dicts_path, PLDDT_THRESHOLD)


if __name__ == "__main__":
    create_patches()
    # trainingDataDir = os.path.join(path.predictionsToDataSetDir, dirName)
    # gridSearchDir = os.path.join(path.aggregateFunctionMLPDir, 'MLP_MSA_val_AUC_stoppage_' + dirName)

    # #
    # trainingDictsDir = os.path.join(trainingDataDir, 'trainingDicts')

    # plotPlddtHistogramForPositivieAndProteome(allPredictions)

    # CREATE PROTEIN OBJECTS, I'M DOING IT IN BATCHES
    # patchesList(allPredictions, int(sys.argv[1]), trainingDataDir, plddtThreshold)

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
