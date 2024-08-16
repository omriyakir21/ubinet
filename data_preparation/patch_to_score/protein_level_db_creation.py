import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import paths
import protein_level_db_creation_utils as protein_db_utils
from data_preparation.ScanNet.db_creation_scanNet_utils import THREE_LETTERS_TO_SINGLE_AA_DICT, load_as_pickle, \
    save_as_pickle
import re


def create_uniprot_names_dict():
    uniprot_names_dict = dict()
    uniprot_names_dict['ubiquitinBinding'] = protein_db_utils.get_uniprot_ids_from_gpad_file(
        os.path.join(paths.ubiquitin_binding_path, 'ubiquitinBinding.gpad'))
    uniprot_names_dict['E1'] = protein_db_utils.get_uniprot_ids_util(os.path.join(paths.E1_path, 'E1_new.txt'))
    uniprot_names_dict['E2'] = protein_db_utils.get_uniprot_ids_util(os.path.join(paths.E2_path, 'E2_new.txt'))
    uniprot_names_dict['E3'] = protein_db_utils.get_uniprot_ids_util(os.path.join(paths.E3_path, 'E3_new.txt'))
    uniprot_names_dict['DUB'] = protein_db_utils.get_uniprot_ids_util(os.path.join(paths.DUB_path, 'DUB_new.txt'))
    protein_db_utils.save_as_pickle(uniprot_names_dict,
                                    os.path.join(paths.GO_source_patch_to_score_path, 'uniprotNamesDictNew.pkl'))


def fetch_af_models_from_user_args(uniprot_names_dict):
    class_name = sys.argv[1]
    i = int(sys.argv[2])
    j = int(sys.argv[3])
    protein_db_utils.fetch_af_models(uniprot_names_dict, class_name, i, j)


def create_evidence_dict():
    evidence_dict = dict()
    evidence_dict['ubiquitinBinding'] = protein_db_utils.get_evidence_from_gpad_file(
        os.path.join(paths.ubiquitin_binding_path, 'ubiquitinBinding.gpad'))
    evidence_dict['E1'] = protein_db_utils.get_evidence_util(os.path.join(paths.E1_path, 'E1_new.txt'))
    evidence_dict['E2'] = protein_db_utils.get_evidence_util(os.path.join(paths.E2_path, 'E2_new.txt'))
    evidence_dict['E3'] = protein_db_utils.get_evidence_util(os.path.join(paths.E3_path, 'E3_new.txt'))
    evidence_dict['DUB'] = protein_db_utils.get_evidence_util(os.path.join(paths.DUB_path, 'DUB_new.txt'))
    protein_db_utils.save_as_pickle(evidence_dict,
                                    os.path.join(paths.GO_source_patch_to_score_path, 'evidence_dict.pkl'))


def rename_AFDB_files():
    # Define the regex pattern to match the filenames
    pattern = re.compile(r'AF-(\w+)-F1-model_v4\.(\w+)')

    # Iterate over each subdirectory in the AFDB directory
    for subdir in os.listdir(paths.AFDB_source_patch_to_score_path):
        subdir_path = os.path.join(paths.AFDB_source_patch_to_score_path, subdir)

        if os.path.isdir(subdir_path):
            # Iterate over each file in the subdirectory
            for filename in os.listdir(subdir_path):
                match = pattern.match(filename)

                if match:
                    uniprot_id, file_type = match.groups()
                    new_filename = f"{uniprot_id}.{file_type}"
                    old_file_path = os.path.join(subdir_path, filename)
                    new_file_path = os.path.join(subdir_path, new_filename)

                    # Rename the file
                    os.rename(old_file_path, new_file_path)
                    print(f"Renamed: {old_file_path} to {new_file_path}")


def save_all_valid_af_predictions_for_all_classes(plddt_threshold, number_of_residues_threshold,
                                                  plddt_ratio_threshold):
    # Define the class names from NEGATIVE and POSITIVE sources
    class_names = protein_db_utils.NEGATIVE_DIRS + protein_db_utils.POSITIVE_DIRS

    # Iterate over each class name and call the function
    for class_name in class_names:
        protein_db_utils.save_all_valid_af_predictions_for_type(
            class_name, plddt_threshold, number_of_residues_threshold,
            plddt_ratio_threshold
        )


def create_csv_file_with_uniprots_info():
    uniprotsDict = dict()
    for class_name in protein_db_utils.NEGATIVE_DIRS:
        uniprotsDict[class_name] = load_as_pickle(
            os.path.join(paths.AFDB_source_patch_to_score_path, class_name, f'allValidOf{class_name}.pkl'))
    for class_name in protein_db_utils.POSITIVE_DIRS:
        uniprotsDict[class_name] = load_as_pickle(
            os.path.join(paths.GO_source_patch_to_score_path, class_name, f'allValidOf{class_name}.pkl'))
    protein_db_utils.write_dict_to_csv(
        os.path.join(paths.patch_to_score_dataset_path, "uniprots_names.csv."), uniprotsDict)


if __name__ == "__main__":
    # create_uniprot_names_dict()
    # create_evidence_dict()

    # evidence_dict = protein_db_utils.load_as_pickle(
    #     os.path.join(paths.GO_source_patch_to_score_path, 'evidence_dict.pkl'))
    # uniprot_names_dict = protein_db_utils.load_as_pickle(
    #     os.path.join(paths.GO_source_patch_to_score_path, 'uniprotNamesDictNew.pkl'))

    # uniprotNames_evidences_list = protein_db_utils.create_uniprot_id_evidence_tuples_for_existing_examples(
    #     uniprot_names_dict, evidence_dict)
    # protein_db_utils.save_as_pickle(uniprotNames_evidences_list,
    #                                 os.path.join(paths.GO_source_patch_to_score_path,
    #                                              'uniprotNames_evidences_list.pkl'))
    # fetch_af_models_from_user_args(uniprot_names_dict)
    # rename_AFDB_files()
    PLDDT_RATIO_THRESHOLD = 0.2
    PLDDT_THRESHOLD = 90
    NUMBER_OF_RESIDUES_THRESHOLD = 100
    save_all_valid_af_predictions_for_all_classes(PLDDT_THRESHOLD, NUMBER_OF_RESIDUES_THRESHOLD,
                                                  PLDDT_RATIO_THRESHOLD)

    # uniprots = getAllUniprotsForTraining(
    #     os.path.join(path.aggregateFunctionMLPDir, os.path.join('dataForTraining2902', 'allInfoDicts.pkl')))
    #
    # uniprotEvidenceDict = uniprotsEvidencesListTodict(uniprotNames_evidences_list)
    #

