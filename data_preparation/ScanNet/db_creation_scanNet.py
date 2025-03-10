import copy
import os
import sys
import shutil
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import pickle

import numpy as np

from Bio import pairwise2
from Bio.PDB import PDBList
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.SASA import ShrakeRupley
from scipy.sparse.csgraph import connected_components
import db_creation_scanNet_utils as db_utils
import paths

UBIQ_LIST_PATH = os.path.join(paths.blast_search_path, "ubiquitin_containing_pdb_entries.txt")


def download_assemblies_and_assymetrics(PDB_names_list):
    '''
    :param UBIQ_LIST_PATH: path to the file containing the list of PDB names
    :return: list of PDB names
    '''
    # download the asymetric files
    pdb_list_object = PDBList()
    asymetricPaths = db_utils.download_asymetric_files(pdb_list_object, PDB_names_list, paths.pdbs_path)
    # download the assemblies
    assemblyPathsLists = db_utils.download_assembly_files(PDB_names_list, pdb_list_object, paths.assemblies_path)
    db_utils.download_assembly_files(PDB_names_list, pdb_list_object, paths.assemblies_path)
    return


def create_save_list_of_entry_dicts():
    '''
    :param listOfPDBNames: list of PDB names
    :return:
    '''
    asymetric_paths, assembly_paths_lists, pdb_names_list = db_utils.order_paths_lists(paths.pdbs_path,
                                                                                       paths.assemblies_path)

    parser = db_utils.parser  # create parser object
    structures = [parser.get_structure(pdb_names_list[i], asymetric_paths[i]) for i in
                  range(len(pdb_names_list))]
    UBD_candidates_list = [db_utils.UBD_candidate(structure) for structure in structures]
    valid_UBD_candidates, valid_PDB_names = db_utils.keep_valid_candidates(UBD_candidates_list, pdb_names_list)
    validAssemblyPathsLists = db_utils.keep_valid_assemblies(valid_PDB_names, assembly_paths_lists)
    list_of_entry_dicts = db_utils.create_list_of_entry_dicts(parser, valid_UBD_candidates, valid_PDB_names,
                                                              validAssemblyPathsLists)
    db_utils.save_as_pickle(list_of_entry_dicts, os.path.join(paths.entry_dicts_path, 'list_of_entry_dicts.pkl'))


def run_create_db_with_user_argv(chosen_assemblies_path, num_sublists):
    # retrive the list of PDB names
    ubiq_path = os.path.join(paths.pdbs_path, '1fxt.cif')
    ubiq_structure = db_utils.parser.get_structure('1FXT', ubiq_path)
    ubiq_chain = ubiq_structure[0]['B']
    ubiq_amino_acids = db_utils.aa_out_of_chain(ubiq_chain)
    ubiq_atoms = db_utils.get_atoms_of_amino_acids(ubiq_chain)
    # ubiqDiameter = db_utils.calculate_diameter(ubiq_atoms)
    ubiq_residues_list = [
        db_utils.THREE_LETTERS_TO_SINGLE_AA_DICT[str(aminoAcid.get_resname())] + str(aminoAcid.get_id()[1]) for
        aminoAcid
        in ubiq_amino_acids]
    chosen_assemblies = db_utils.load_as_pickle(chosen_assemblies_path)
    chosenAssembliesListOfSublists = db_utils.split_list(chosen_assemblies, num_sublists)
    items = [(chosenAssembliesListOfSublists[i], i) for i in range(num_sublists)]
    db_utils.create_data_base(items[int(sys.argv[1])], ubiq_residues_list)  # download
    # db_utils.create_data_base(items[12], ubiq_residues_list)

def integrate_checkchains_per_batch(batch_dir, prefix_batch, prefix_file, num_sublists):
    for i in range(num_sublists):
        batch_path = os.path.join(batch_dir, f"{prefix_batch}{i}")
        integrated_content = ""
        for j in range(1, 25):
            file_path = os.path.join(batch_path, f"{prefix_file}_{j}_mer.txt")
            if os.path.exists(file_path):
                with open(file_path, 'r') as file:
                    integrated_content += file.read()
        integrated_file_path = os.path.join(batch_path, f"Integrated_{prefix_file}_mer.txt")
        with open(integrated_file_path, 'w') as integrated_file:
            integrated_file.write(integrated_content)


def integrate_all_batches_checkchains(batch_dir, output_file, prefix_batch, prefix_file, num_sublists):
    integrated_content = ""
    for i in range(num_sublists):
        batch_path = os.path.join(batch_dir, f"{prefix_batch}{i}")
        integrated_file_path = os.path.join(batch_path, f"Integrated_{prefix_file}_mer.txt")
        if os.path.exists(integrated_file_path):
            with open(integrated_file_path, 'r') as file:
                integrated_content += file.read()
    with open(output_file, 'w') as output:
        output.write(integrated_content)


def integrate_all_batches_summarylog(batch_dir, output_file, num_sublists):
    integrated_content = ""
    for i in range(num_sublists):
        batch_path = os.path.join(batch_dir, f"Batch{i}")
        summary_log_path = os.path.join(batch_path, "summaryLog.txt")
        if os.path.exists(summary_log_path):
            with open(summary_log_path, 'r') as file:
                integrated_content += '\n'
                integrated_content += file.read()
    with open(output_file, 'w') as output:
        output.write(integrated_content)

def integrate_all_files(num_sublists):
    integrate_checkchains_per_batch(paths.ImerFiles_path, 'Batch', 'Checkchains', num_sublists)
    integrate_checkchains_per_batch(paths.ASA_path, 'asaBatch', 'Checkchains_asa', num_sublists)
    integrate_all_batches_checkchains(paths.ImerFiles_path,
                                      os.path.join(paths.ImerFiles_path, 'Integrated_Checkchains_mer.txt'), 'Batch',
                                      'Checkchains', num_sublists)
    integrate_all_batches_checkchains(paths.ASA_path,
                                      os.path.join(paths.ASA_path, 'Integrated_Checkchains_asa_mer.txt'), 'asaBatch',
                                      'Checkchains_asa', num_sublists)
    integrate_all_batches_summarylog(paths.ImerFiles_path, os.path.join(paths.ImerFiles_path, 'Integrated_summaryLog.txt'), num_sublists)


def copy_files_to_new_folder(file_paths, new_folder_path):
    """
    Copies a list of files to a new folder.

    :param file_paths: List of file paths to be copied.
    :param new_folder_path: Path to the new folder where files will be copied.
    """
    # Create the new folder if it doesn't exist
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)

    # Copy each file to the new folder
    for file_path in file_paths:
        if os.path.isfile(file_path):
            shutil.copy(file_path, new_folder_path)
        else:
            print(f"File not found: {file_path}")


def rename_cif_files(folder_path):
    """
    Renames .cif files in the specified folder from {pdb_name}-{some string}.cif to {pdb_name}.cif.

    :param folder_path: Path to the folder containing the .cif files.
    """
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.cif'):
            parts = file_name.split('-')
            if len(parts) > 1:
                pdb_name = parts[0]
                new_file_name = f"{pdb_name}.cif"
                old_file_path = os.path.join(folder_path, file_name)
                new_file_path = os.path.join(folder_path, new_file_name)
                os.rename(old_file_path, new_file_path)
                print(f"Renamed {file_name} to {new_file_name}")


if __name__ == "__main__":
    # retrive the list of PDB names
    # PDB_names_list = db_utils.read_PDB_names_from_file(UBIQ_LIST_PATH)
    # download the assemblies and assymetrics files
    # download_assemblies_and_assymetrics(PDB_names_list)
    # create_save_list_of_entry_dicts()
    # chosen_assemblies = db_utils.from_pickle_to_choose_assemblies(
    #     os.path.join(paths.entry_dicts_path, 'list_of_entry_dicts_with_probabilities.pkl'))
    NUM_SUBLISTS = 40
    chosen_assemblies = db_utils.load_as_pickle(os.path.join(paths.assemblies_path, 'chosen_assemblies.pkl'))
    print(len(chosen_assemblies))
    run_create_db_with_user_argv(os.path.join(paths.assemblies_path, 'chosen_assemblies.pkl'), NUM_SUBLISTS)
    integrate_all_files(NUM_SUBLISTS)  
    # copy_files_to_new_folder(chosen_assemblies, paths.chosen_assemblies_path)
    # rename_cif_files(paths.chosen_assemblies_path)
    integrate_all_batches_summarylog(paths.ImerFiles_path, os.path.join(paths.ImerFiles_path, 'Integrated_summaryLog2.txt'), NUM_SUBLISTS)