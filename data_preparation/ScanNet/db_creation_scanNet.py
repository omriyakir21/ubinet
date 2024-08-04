import copy
import os
import sys

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


if __name__ == "__main__":
    # retrive the list of PDB names
    # PDB_names_list = db_utils.read_PDB_names_from_file(UBIQ_LIST_PATH)
    # download the assemblies and assymetrics files
    # download_assemblies_and_assymetrics(PDB_names_list)
    # create_save_list_of_entry_dicts()

    ubiq_path = os.path.join(paths.pdbs_path, '3by4.cif')
    ubiq_structure = db_utils.parser.get_structure('3BY4', ubiq_path)
    ubiq_chain = ubiq_structure[0]['B']  # uni-prot = "UBIQ-HUMAN"
    ubiq_seq = db_utils.get_str_seq_of_chain(ubiq_chain)
    ubiq_amino_acids = db_utils.aa_out_of_chain(ubiq_chain)
    ubiq_atoms = db_utils.get_atoms_of_amino_acids(ubiq_chain)
    ubiqDiameter = db_utils.calculate_diameter(ubiq_atoms)
    ubiq_residues_list = [
        db_utils.THREE_LETTERS_TO_SINGLE_AA_DICT[str(aminoAcid.get_resname())] + str(aminoAcid.get_id()[1]) for
        aminoAcid
        in ubiq_amino_acids]

    # list_of_entry_dicts = db_utils.load_as_pickle(os.path.join(paths.entry_dicts_path, 'list_of_entry_dicts.pkl'))
    # db_utils.pickle_list_of_entry_dicts(list_of_entry_dicts, 852)
    # Queen_predictions_path = db_utils.load_as_pickle(os.path.join(paths.QUEEN_path, 'predictions'))

    chosen_assemblies = db_utils.from_pickle_to_choose_assemblies(
        os.path.join(paths.entry_dicts_path, 'list_of_entry_dicts_with_probabilities.pkl'))
    # chosenAssembliesListOfSublists = db_utils.split_list(chosen_assemblies, 40)
    # items = [(chosenAssembliesListOfSublists[i], i) for i in range(40)]
    # createDataBase(items[14])
