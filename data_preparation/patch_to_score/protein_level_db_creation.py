import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import paths
import protein_level_db_creation_utils as protein_db_utils
from data_preparation.ScanNet.db_creation_scanNet_utils import THREE_LETTERS_TO_SINGLE_AA_DICT

def create_uniprot_names_dict():
    uniprot_names_dict = dict()
    uniprot_names_dict['ubiquitinBinding'] = protein_db_utils.get_uniprot_ids_from_gpad_file(
        os.path.join(paths.ubiquitin_binding_path, 'ubiquitinBinding.gpad'))
    uniprot_names_dict['E1'] = protein_db_utils.get_uniprot_ids_util(os.path.join(paths.E1_path, 'E1_new.txt'))
    uniprot_names_dict['E2'] = protein_db_utils.get_uniprot_ids_util(os.path.join(paths.E2_path, 'E2_new.txt'))
    uniprot_names_dict['E3'] = protein_db_utils.get_uniprot_ids_util(os.path.join(paths.E3_path, 'E3_new.txt'))
    uniprot_names_dict['DUB'] = protein_db_utils.get_uniprot_ids_util(os.path.join(paths.DUB_path, 'DUB_new.txt'))
    protein_db_utils.save_as_pickle(uniprot_names_dict,
                                    os.path.join(paths.GO_source_patch_to_score_path, 'uniprotNamesDictNew.pkl')

def create_evidence_dict():
    evidence_dict = dict()
    evidence_dict['ubiquitinBinding'] = protein_db_utils.get_evidence_from_gpad_file(
        os.path.join(paths.ubiquitin_binding_path, 'ubiquitinBinding.gpad'))
    evidence_dict['E1'] = protein_db_utils.get_evidence_util(os.path.join(paths.E1_path, 'E1_new.txt'))
    evidence_dict['E2'] = protein_db_utils.get_evidence_util(os.path.join(paths.E2_path, 'E2_new.txt'))
    evidence_dict['E3'] = protein_db_utils.get_evidence_util(os.path.join(paths.E3_path, 'E3_new.txt'))
    evidence_dict['DUB'] = protein_db_utils.get_evidence_util(os.path.join(paths.DUB_path, 'DUB_new.txt'))
    protein_db_utils.save_as_pickle(evidence_dict, os.path.join(paths.GO_source_patch_to_score_path, 'evidence_dict'))


if __name__ == "__main__":
    create_uniprot_names_dict()
    create_evidence_dict()

    # evidence_dict = loadPickle(os.path.join(path.GoPath, 'evidence_dict.pkl'))
    # uniprotNamesDict = loadPickle(os.path.join(path.GoPath, 'uniprotNamesDictNew.pkl'))
    # uniprotNames_evidences_list = createUniprotId_EvidenceTuplesForExistingExamples(uniprotNamesDict, evidence_dict)
    # saveAsPickle(uniprotNames_evidences_list,
    #              r'C:\Users\omriy\UBDAndScanNet\newUBD\UBDModel\GO\uniprotNames_evidences_list')
    # uniprots = getAllUniprotsForTraining(
    #     os.path.join(path.aggregateFunctionMLPDir, os.path.join('dataForTraining2902', 'allInfoDicts.pkl')))
    #
    # uniprotEvidenceDict = uniprotsEvidencesListTodict(uniprotNames_evidences_list)
    #
    # fetchAFModels(uniprotNamesDict, 'E2')

# uniprotNamesDict = loadPickle(r'C:\Users\omriy\UBDAndScanNet\UBDModel\GO\uniprotNamesDict.pkl')
# directory_path =r'C:\Users\omriy\UBDAndScanNet\UBDModel\GO\E3'
# files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
# file_count = len(files)
# print(file_count)
# fetchAFModels(uniprotNamesDict, 'E3', 16000, 18582)
# print(len(uniprotNamesDict['DUB']))
# print(len(uniprotNamesDict['ubiquitinBinding']))
# fetchAFModels(uniprotNamesDict, 'ubiquitinBinding', 72000, 831100)
# fetchAFModels(uniprotNamesDict, 'DUB', 2000, 4000)
# fetchAFModels(uniprotNamesDict, 'DUB', 8000, 8800)
# fetchAFModels(uniprotNamesDict, 'E3', 12000, 14000)
# fetchAFModels(uniprotNamesDict, 'E3', 14000, 16000)
# fetchAFModels(uniprotNamesDict, 'E3', 16000, 18582)


# pdbFilePath = r'C:\Users\omriy\UBDAndScanNet\UBDModel\GO\E1\A0A0A0KE12.cif'
# name = 'A0A0A0KE12'


# isValidAFPrediction(pdbFilePath,name)


# getAllValidAFPredictionsForType('ubiquitinBinding')
# f = loadPickle('GO/allValidOfE1.pkl')
# print(1)


# getAllValidAFPredictionsForType('E3')
# uniprotNamesDict['ubiquitinBinding'] = uniprotNamesOfGpadfile(
#     r'C:\Users\omriy\UBDAndScanNet\UBDModel\GO\ubiquitinBinding.gpad')
# saveAsPickle(uniprotNamesDict, r'C:\Users\omriy\UBDAndScanNet\UBDModel\GO\uniprotNamesDictAll')
# f = loadPickle(r'C:\Users\omriy\UBDAndScanNet\UBDModel\GO\uniprotNamesDictAll.pkl')
# print(1)
# getAllValidAFPredictionsForType('ubiquitinBinding')


# fileName = 'nameSeqTuples' + 'proteome'
# saveAsPickle(getUniprotSequenceTuplesForType('proteome'), fileName)


# GOPath = '/mnt/c/Users/omriy/UBDAndScanNet/UBDModel/GO'


# allValidOfDUB = loadPickle(os.path.join(GOPath, 'allValidOfDUB.pkl'))
# allValidOfE1 = loadPickle(os.path.join(GOPath, 'allValidOfE1.pkl'))
# allValidOfE2 = loadPickle(os.path.join(GOPath, 'allValidOfE2.pkl'))
# allValidOfE3 = loadPickle(os.path.join(GOPath, 'allValidOfE3.pkl'))
# allValidOfubiquitinBinding = loadPickle(os.path.join(GOPath, 'allValidOfubiquitinBinding.pkl'))
#
# nameSeqTuplesE1 = loadPickle(os.path.join(GOPath, 'nameSeqTuplesE1.pkl'))
# nameSeqTuplesE2 = loadPickle(os.path.join(GOPath, 'nameSeqTuplesE2.pkl'))
# nameSeqTuplesE3 = loadPickle(os.path.join(GOPath, 'nameSeqTuplesE3.pkl'))
# nameSeqTuplesDUB = loadPickle(os.path.join(GOPath, 'nameSeqTuplesDUB.pkl'))
# nameSeqTuplesubiquitinBinding = loadPickle(os.path.join(GOPath, 'nameSeqTuplesubiquitinBinding.pkl'))
# nameSeqTuplesProteome = loadPickle(os.path.join(GOPath, 'nameSeqTuplesproteome.pkl'))


# seqListPositives = CreateAllTypesListOfSequences([nameSeqTuplesE1, nameSeqTuplesE2, nameSeqTuplesE3, nameSeqTuplesDUB,
#                        nameSeqTuplesubiquitinBinding])
# saveAsPickle(seqListPositives, r'C:\Users\omriy\UBDAndScanNet\UBDModel\GO\seqListPositives')


# createPathsTextFileForType('ubiquitinBinding')


# seqListNegatives = [tup[1] for tup in nameSeqTuplesProteome]
#
# ubuntu = False
#
# if ubuntu:
#     Path = '/mnt/c/Usersroot/omriy/UBDAndScanNet'
# path2mmseqstmp = '/mnt/c/Users/omriy/UBDAndScanNet/UBDModel/mmseqs2'
# cluster_indices, representative_indices = cluster_sequences(seqListNegatives,path2mmseqstmp, seqid=0.7, coverage=0.8, covmode='0')
#     saveAsPickle(cluster_indices, '/mnt/c/Users/omriy/UBDAndScanNet/UBDModel/GO/ClusterIndicesNegatives70')
#     saveAsPickle(representative_indices,
#                  '/mnt/c/Users/omriy/UBDAndScanNet/UBDModel/GO/representativeIndicesNegatives70')
# else:
#     rootPath = r'C:\Users\omriy\UBDAndScanNet'
#     seqListPositives = loadPickle(os.path.join(GOPath, 'seqListPositives.pkl'))


# clusterIndexes = loadPickle(rootPath + 'UBDModel/mmseqs2/clusterIndices.pkl')

# ClusterIndicesPositives = loadPickle((os.path.join(GOPath, 'ClusterIndicesPositives95.pkl')))
# representativeIndicesPositives = loadPickle((os.path.join(GOPath, 'representativeIndicesPositives95.pkl')))
# path2mafft = '/usr/bin/mafft'
# indexesDict = dict()
# indexesDict['E1'] = (0, len(nameSeqTuplesE1))
# indexesDict['E2'] = (indexesDict['E1'][1], indexesDict['E1'][1] + len(nameSeqTuplesE2))
# indexesDict['E3'] = (indexesDict['E2'][1], indexesDict['E2'][1] + len(nameSeqTuplesE3))
# indexesDict['DUB'] = (indexesDict['E3'][1], indexesDict['E3'][1] + len(nameSeqTuplesDUB))
# indexesDict['ubiquitinBinding'] = (indexesDict['DUB'][1], indexesDict['DUB'][1] + len(nameSeqTuplesubiquitinBinding))


# ClusterIndicesNegatives = loadPickle(os.path.join(GOPath, 'ClusterIndicesNegatives95.pkl'))
#
# E1NameList = createNameList(nameSeqTuplesE1)
# E2NameList = createNameList(nameSeqTuplesE2)
# E3NameList = createNameList(nameSeqTuplesE3)
# DUBNameList = createNameList(nameSeqTuplesDUB)
# ubiquitinBindingNameList = createNameList(nameSeqTuplesubiquitinBinding)
# proteomeNameList = createNameList(nameSeqTuplesProteome)
#
# nameClusterTypeDictPositives = dict()
# nameClusterTypeDictNegatives = dict()
# creatNameClusterTypeDict(ClusterIndicesPositives, E1NameList, indexesDict['E1'][0], indexesDict['E1'][1],
#                          nameClusterTypeDictPositives, 'E1')
# creatNameClusterTypeDict(ClusterIndicesPositives, E2NameList, indexesDict['E2'][0], indexesDict['E2'][1],
#                          nameClusterTypeDictPositives, 'E2')
# creatNameClusterTypeDict(ClusterIndicesPositives, E3NameList, indexesDict['E3'][0], indexesDict['E3'][1],
#                          nameClusterTypeDictPositives, 'E3')
# creatNameClusterTypeDict(ClusterIndicesPositives, DUBNameList, indexesDict['DUB'][0], indexesDict['DUB'][1],
#                          nameClusterTypeDictPositives, 'DUB')
# creatNameClusterTypeDict(ClusterIndicesPositives, ubiquitinBindingNameList, indexesDict['ubiquitinBinding'][0],
#                          indexesDict['ubiquitinBinding'][1],
#                          nameClusterTypeDictPositives, 'ubiquitinBinding')
# creatNameClusterTypeDict(ClusterIndicesNegatives, proteomeNameList, 0, len(proteomeNameList),
#                          nameClusterTypeDictNegatives, 'Proteome')
# saveAsPickle(nameClusterTypeDictPositives, 'nameClusterTypeDictPositives')
# saveAsPickle(nameClusterTypeDictNegatives, 'nameClusterTypeDictNegatives')


# createTargetsFileForType('proteome')


# input_file = r'C:\Users\omriy\UBDAndScanNet\ScanNet_Ub\tagetsubiquitinBinding.txt' # Replace with your input file name
# output_prefix = r'C:\Users\omriy\UBDAndScanNet\ScanNet_Ub\targetsubiquitinBinding\targetsubiquitinBinding'
# lines_per_file = 700
#
# split_file(input_file, output_prefix, lines_per_file)


# uniprotsDict = dict()
# uniprotsDict['E1'] = GetAllValidUniprotIdsOfType('E1')
# uniprotsDict['E2'] = GetAllValidUniprotIdsOfType('E2')
# uniprotsDict['E3'] = GetAllValidUniprotIdsOfType('E3')
# uniprotsDict['DUB'] = GetAllValidUniprotIdsOfType('DUB')
# uniprotsDict['ubiquitinBinding'] = GetAllValidUniprotIdsOfType('ubiquitinBinding')
# uniprotsDict['proteome'] = GetAllValidUniprotIdsOfType('proteome', True)


# write_dict_to_csv(os.path.join(r'C:\Users\omriy\UBDAndScanNet\UBDModel\GO', "uniprotnamecsCSV.csv."), uniprotsDict)


# proteomeUniprots = GetAllValidUniprotIdsOfType('proteome', True)
# proteomeEvidences = [getEvidenceOfUniprotId(uniprotId) for uniprotId in proteomeUniprots]
# proteome_uniprotNames_evidences_list = [(proteomeUniprots[i], proteomeEvidences[i]) for i in
#                                         range(len(proteomeUniprots))]
# assert len(proteomeUniprots) == len(proteomeEvidences)
# proteomeUniprotEvidenceDict = uniprotsEvidencesListTodict(proteome_uniprotNames_evidences_list)
# saveAsPickle(proteomeUniprotEvidenceDict,
#              os.path.join(path.GoPath, os.path.join('proteome', 'proteomeUniprotEvidenceDict')))

# print(sys.argv[0])
# nonProteomeUniprots = load_as_pickle(
#     os.path.join(path.aggregateFunctionMLPDir, os.path.join('gridSearch11_3', 'allUniprotsExceptProteome.pkl')))
# nonProteomeUniprotsSplitted = (
#     nonProteomeUniprots[:len(nonProteomeUniprots) // 2], nonProteomeUniprots[len(nonProteomeUniprots) // 2:])
# nonProteomeUniprots = nonProteomeUniprotsSplitted[int(sys.argv[1])]
# nonProteomeEvidences = [get_evidence_of_uniprot_id(uniprotId) for uniprotId in nonProteomeUniprots]
# proteome_uniprotNames_evidences_list = [(nonProteomeUniprots[i], nonProteomeEvidences[i]) for i in
#                                         range(len(nonProteomeUniprots))]
# assert len(nonProteomeUniprots) == len(nonProteomeEvidences)
# proteomeUniprotEvidenceDict = uniprots_evidences_list_todict(proteome_uniprotNames_evidences_list)
# save_as_pickle(proteomeUniprotEvidenceDict,
#                os.path.join(path.GoPath, 'nonProteomeUniprotEvidenceDict' + sys.argv[1]))

# nonProteomeUniprots = loadPickle(
#     os.path.join(path.aggregateFunctionMLPDir, os.path.join('gridSearch11_3', 'allUniprotsExceptProteome.pkl')))
# nonProteomeUniprotsSplitted = (
# nonProteomeUniprots[:len(nonProteomeUniprots) // 2], nonProteomeUniprots[len(nonProteomeUniprots) // 2:])
# nonProteomeUniprots = nonProteomeUniprotsSplitted[int(sys.argv[1])]
# nonProteomeEvidences = [getEvidenceOfUniprotId(uniprotId) for uniprotId in nonProteomeUniprots]
# proteome_uniprotNames_evidences_list = [(nonProteomeUniprots[i], nonProteomeEvidences[i]) for i in
#                                         range(len(nonProteomeUniprots))]
# assert len(nonProteomeUniprots) == len(nonProteomeEvidences)
# proteomeUniprotEvidenceDict = uniprotsEvidencesListTodict(proteome_uniprotNames_evidences_list)
# saveAsPickle(proteomeUniprotEvidenceDict,
#              os.path.join(path.GoPath, 'nonProteomeUniprotEvidenceDict' + sys.argv[1]))