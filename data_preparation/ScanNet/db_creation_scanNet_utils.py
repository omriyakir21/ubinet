import copy
import os
import pickle
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from Bio import pairwise2
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.SASA import ShrakeRupley
from scipy.sparse.csgraph import connected_components
import numpy as np
import traceback

import paths

THREE_LETTERS_TO_SINGLE_AA_DICT = {'GLY': 'G', 'ALA': 'A', 'VAL': 'V', 'LEU': 'L', 'ILE': 'I', 'THR': 'T', 'SER': 'S',
                                   'MET': 'M', 'CYS': 'C', 'PRO': 'P', 'PHE': 'F', 'TYR': 'Y', 'TRP': 'W', 'HIS': 'H',
                                   'LYS': 'K', 'ARG': 'R', 'ASP': 'D', 'GLU': 'E',
                                   'ASN': 'N', 'GLN': 'Q'}
UBIQ_SEQ = 'MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQRESTLHLVLRLRGG'

INV_MAP = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 10, 9: 12, 10: 14,
           11: 24}  # key=index in Queen value = number of units in multimer

OPPOSITE_MAP = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 10: 8, 12: 9, 14: 10, 24: 11}

parser = MMCIFParser()


def save_as_pickle(obj, file_path):
    """
    Saves an object to a file using pickle.

    Parameters:
    obj (any): The object to be saved.
    file_path (str): The path to the file where the object should be saved.
    """
    with open(file_path, 'wb') as file:
        pickle.dump(obj, file)


def load_as_pickle(file_path):
    """
    Loads an object from a file using pickle.

    Parameters:
    file_path (str): The path to the file where the object is saved.

    Returns:
    any: The object loaded from the file.
    """
    with open(file_path, 'rb') as file:
        return pickle.load(file)


def read_PDB_names_from_file(path):
    """
    :param path: path of a file containing a coma separated PDB ID's
    :return: list containing all PDB names from the txt file
    """
    pdb_names_list = []
    PDB_text_file = open(path, "r")
    PDB_lines = PDB_text_file.readlines()
    for line in PDB_lines:
        for id in line.split(","):
            pdb_names_list.append(id)
    return pdb_names_list


def download_assembly_files(pdb_names_list, pdb_list_object, dir_path):
    """
    :param pdb_names_list: list of pdb names
    :param pdb_list_object: pdbList Object
    :param dir_path: directory path to add all assemblies
    :return: list of lists- for every pdb name all of the assembly file names
    """
    assembly_paths_list = [[] for _ in range(len(pdb_names_list))]
    for i in range(len(pdb_names_list)):
        pdb_name = pdb_names_list[i]
        new_dir_path = dir_path + "/" + pdb_name
        if not os.path.exists(new_dir_path):
            os.mkdir(new_dir_path)
        assembly_num = 1
        while True:
            assembly_path = pdb_list_object.retrieve_assembly_file(pdb_name, assembly_num=assembly_num, pdir=new_dir_path,
                                                                  file_format="mmCif")
            if os.path.exists(assembly_path):
                assembly_paths_list[i].append(assembly_path)
                assembly_num += 1
            else:
                break
    return assembly_paths_list


def download_asymetric_files(pdb_list_object, pdb_names_list, dir_path):
    file_names = pdb_list_object.download_pdb_files(pdb_codes=pdb_names_list, overwrite=True, file_format="mmCif",
                                                   pdir=dir_path)
    return file_names


def redownload_failed_assemblies(PDB_names_list, pdb_list_object, dir_path):
    for i in range(len(PDB_names_list)):
        pdb_name = PDB_names_list[i]
        new_dir_path = dir_path + "/" + pdb_name
        num_files = len(os.listdir(new_dir_path))
        if num_files == 0:  # failed
            assembly_num = 1
            while True:
                assemblyPath = pdb_list_object.retrieve_assembly_file(pdb_name, assembly_num=assembly_num,
                                                                      pdir=new_dir_path,
                                                                      file_format="mmCif")
                if os.path.exists(assemblyPath):
                    assembly_num += 1
                else:
                    break


def create_assembly_paths_lists(assemblies_dir_path):
    """
    :param assemblies_dir_path: path to the directory containing the assemblies
    :return: assembly_paths_lists where assembly_paths_lists[i] is a list containing all the assembly paths of the i'th pdb structure
    """
    assembly_paths_lists = []
    assembly_names = []
    for pdbDir in os.listdir(assemblies_dir_path):
        #make sure its a directory
        if not os.path.isdir(os.path.join(assemblies_dir_path, pdbDir)):
            continue
        assembly_names.append(pdbDir.lower())
        assembly_paths_list = []
        pdb_dir_path = os.path.join(assemblies_dir_path, pdbDir)
        for assembly_path in os.listdir(pdb_dir_path):
            assembly_path = os.path.join(pdb_dir_path, assembly_path)
            assembly_paths_list.append(assembly_path)
        assembly_paths_lists.append(assembly_paths_list)
    return assembly_paths_lists, assembly_names


def create_asymetric_paths_list(pdbs_path):
    """
    :param pdbs_path: path to the directory containing the asymetric files
    :return:asymetric_paths - a list containing the paths pdb_path
    """
    asymetric_names = []
    asymetric_paths = []
    for pdb_path in os.listdir(pdbs_path):
        pdb_dir_path = os.path.join(pdbs_path, pdb_path)
        asymetric_paths.append(pdb_dir_path)
        asymetric_names.append(pdb_path.split('.')[0])
    return asymetric_paths, asymetric_names


def order_paths_lists(pdbs_path, assemblies_path):
    ordered_asymmetric_paths = []
    ordered_assembly_paths_lists = []
    asymetric_paths, asymetric_names = create_asymetric_paths_list(pdbs_path)
    assembly_paths_lists, assembly_names = create_assembly_paths_lists(assemblies_path)
    ordered_pdb_name_list = [name for name in asymetric_names if name in assembly_names]
    for i in range(len(ordered_pdb_name_list)):
        index = asymetric_names.index(ordered_pdb_name_list[i])
        ordered_asymmetric_paths.append(asymetric_paths[index])
        index = assembly_names.index(ordered_pdb_name_list[i])
        ordered_assembly_paths_lists.append(assembly_paths_lists[index])
    return (ordered_asymmetric_paths, ordered_assembly_paths_lists, ordered_pdb_name_list)


def aa_out_of_chain(chain):
    """
    :param chain: chain object
    :return: list of aa (not HOH molecule)
    """
    my_list = []
    amino_acids = chain.get_residues()
    for aa in amino_acids:
        name = str(aa.get_resname())
        if name in THREE_LETTERS_TO_SINGLE_AA_DICT.keys():  # amino acid and not other molecule
            my_list.append(aa)
    return my_list


def get_str_seq_of_chain(chain):
    """
    :param chain: chain
    :return: Its sequence
    """
    listOfAminoAcids = aa_out_of_chain(chain)
    return "".join([THREE_LETTERS_TO_SINGLE_AA_DICT[aa.get_resname()] for aa in listOfAminoAcids])


def calculate_identity(seq_a, seq_b):
    """
    :param seq_a: The sequence of amino acid from chain A
    :param seq_b: The sequence of amino acid from chain B
    :return: percentage of identity between the sequences
    """
    score = pairwise2.align.globalxx(seq_a, seq_b, one_alignment_only=True, score_only=True)
    # min_len = min(len(seqA), len(seqB))
    # identity = score / min_len
    max_len = max(len(seq_a), len(seq_b))
    identity = score / max_len
    return identity


def is_ubiquitin(chain):
    """
    :param chain: a chain
    :return: True iff its a Ubiquitin chain
    """
    seq = get_str_seq_of_chain(chain)
    if len(seq) == 0:  # invalid chain
        return None
    identity_threshold = 0.9
    is_ubiq = identity_threshold < calculate_identity(seq, UBIQ_SEQ)
    return is_ubiq


class Model:
    def classify_model_chains(self):
        """
        :return: The function classifies the model's
        chains to ubiquitin chains or non ubiquitin chains.
        """
        for chain in self.chains:
            is_ubiq = is_ubiquitin(chain)
            if is_ubiq is None:  # chain is invalid
                continue
            if is_ubiq:
                self.ubiq_chains.append(chain)
            else:
                self.non_ubiq_chains.append(chain)

    def is_valid_model(self):
        """
        :return: The function returns True iff there is at list one ubiquitin and non ubiquitin chains in the model
        """
        if len(self.ubiq_chains) > 0 and len(self.non_ubiq_chains) > 0:
            return True
        return False

    def calculate_ASAF(self):
        sasa_calc = ShrakeRupley()

        for i in range(len(self.non_ubiq_chains)):
            aminoAcids = aa_out_of_chain(self.non_ubiq_chains[i])
            for aa in aminoAcids:
                id = aa.get_segid
                sasa_calc.compute(aa, level="R")
                asa = aa.sasa
                self.sasa_results_dict[id] = asa

    def __init__(self, model):
        self.model = model
        self.chains = model.get_chains()
        self.id = model.id
        self.ubiq_chains = []
        self.non_ubiq_chains = []
        self.sasa_results_dict = dict()
        self.classify_model_chains()
        self.calculate_ASAF()


class UBD_candidate:

    def create_models_for_structure(self):
        """
        The function add the structure's models to models field (as Model class object)
        :return: None
        """
        for model in self.structure.get_models():
            my_model = Model(model)
            if my_model.is_valid_model():
                self.models.append(my_model)

    def __init__(self, structure):
        self.structure = structure
        self.models = []
        self.create_models_for_structure()


def atom_dists(atom1, atoms):
    vector1 = atom1.get_coord()
    vectors = np.array([atom.get_coord() for atom in atoms])
    distances = np.sqrt(((vectors - vector1[np.newaxis]) ** 2).sum(-1))
    return distances


def get_atoms_of_chain(chain):
    aminoAcids = aa_out_of_chain(chain)
    return get_atoms_of_amino_acids(aminoAcids)


def get_atoms_of_amino_acids(aminoAcids):
    """
    :param aminoAcids: list of a chain's aminoAcids objects
    :return: list of the chain's atoms
    """
    atoms = []
    for aa in aminoAcids:
        atoms += aa.get_atoms()
    return atoms


def calculate_diameter(atoms):
    globalMaxDistance = 0
    for atom in atoms:
        maxDistance = atom_dists(atom, atoms).max()
        if maxDistance > globalMaxDistance:
            globalMaxDistance = copy.copy(maxDistance)
    return globalMaxDistance


def calculate_diameter_from_chain(chain):
    atoms = get_atoms_of_chain(chain)
    diameter = calculate_diameter(atoms)
    return diameter


def get_corresponding_ubiq_residues(aa_string, ubiq_residus_list):
    assert (len(ubiq_residus_list) == len(UBIQ_SEQ))
    alignments = pairwise2.align.globalxx(aa_string, UBIQ_SEQ)
    alignment1 = alignments[0].seqA
    alignment2 = alignments[0].seqB
    index1 = 0
    index2 = 0
    corresponding_ubiq_residue_list = [None for _ in range(len(aa_string))]
    for i in range(len(alignment1)):
        if alignment2[i] != '-' and alignment1[i] != '-':
            corresponding_ubiq_residue_list[index1] = ubiq_residus_list[index2]
        if alignment1[i] != '-':
            index1 += 1
        if alignment2[i] != '-':
            index2 += 1
        if index1 == len(aa_string) or index2 == len(UBIQ_SEQ):
            break
    return corresponding_ubiq_residue_list


def keep_valid_candidates(UBD_candidates_list, PDB_names_list):
    """
    :param UBD_candidates_list: list of UBD_candidates objects
    :return: List with the valid candidates in the list
    """
    assert (len(UBD_candidates_list) == len(PDB_names_list))
    valid_UBD_candidates = []
    valid_PDB_names = []
    for i in range(len(UBD_candidates_list)):
        # for candidate in UBD_candidates_list:
        candidate = UBD_candidates_list[i]
        if len(candidate.models) > 0:
            valid_UBD_candidates.append(candidate)
            valid_PDB_names.append(PDB_names_list[i])
    return valid_UBD_candidates, valid_PDB_names


def keep_valid_assemblies(valid_PDB_names, assembly_paths_lists):
    """
    :param valid_PDB_names: names of the valid pdbs (has ubiq and non ubiq)
    :param assembly_paths_lists: where assemblyPathsLists[i] is a list containing all the assembly paths of the i'th pdb structure
    :return: valid_assembly_paths_lists which is the assemblies pathsLists of the valid pdbs
    """
    valid_assembly_paths_lists = []
    for i in range(len(assembly_paths_lists)):
        assembly_pdb_name = assembly_paths_lists[i][0].split("/")[-2].lower()
        if assembly_pdb_name in valid_PDB_names:
            valid_assembly_paths_lists.append(assembly_paths_lists[i])
    return valid_assembly_paths_lists


def find_longest_non_ubiq(valid_UBD_candidate):
    """
    :param valid_UBD_candidate: UBD_candidate Object
    :return: The sequence of the longest Non ubiq chain in the structure
    """
    model = valid_UBD_candidate.models[0]  # first model
    max_chain_length = 0
    max_chain_amino_acids = None
    for i in range(len(model.non_ubiq_chains)):
        chain_amino_acids = get_str_seq_of_chain(model.non_ubiq_chains[i])
        if len(chain_amino_acids) >= max_chain_length:
            max_chain_length = len(chain_amino_acids)
            max_chain_amino_acids = chain_amino_acids
    return max_chain_amino_acids


def find_number_of_copies_for_sequence(parser, assembly_path, pdb_name, reference_sequence):
    """
    :param assembly_path: path to assembly file
    :param reference_sequence: string that represents a sequence ( ‘ACDEFGHI’ )
    :return: number of copies the sequence in the assembly structure
    """

    identity_threshold = 0.95
    structure = parser.get_structure(pdb_name, assembly_path)
    number_of_copies = 0
    structure = UBD_candidate(structure)
    if len(structure.models) == 0:
        return None  # assembly is not valid
    model = structure.models[0]
    to_print_list = []
    if assembly_path == 'C:/Users/omriy/pythonProject/ubiq_project/assemblies\\1WR6\\1wr6-assembly4.cif':
        to_print_list.append("reference: " + reference_sequence)
        for non_ubiq_chain in model.non_ubiq_chains:
            to_print_list.append(non_ubiq_chain.full_id[2] + " " + get_str_seq_of_chain(non_ubiq_chain))
    for non_ubiq_chain in model.non_ubiq_chains:
        seq_to_compare = get_str_seq_of_chain(non_ubiq_chain)
        identity = calculate_identity(reference_sequence, seq_to_compare)
        if identity_threshold < identity:
            number_of_copies += 1
    return number_of_copies


def create_entry_dict(parser, index, pdbName, assembly_paths_list, valid_ubd_candidate):
    entryDict = {}
    entryDict['assemblyPathsList'] = assembly_paths_list
    entryDict['index'] = index
    entryDict['entry'] = pdbName
    assemblies = [i + 1 for i in range(len(assembly_paths_list))]
    entryDict['assemblies'] = assemblies
    reference_sequence = find_longest_non_ubiq(valid_ubd_candidate)
    entryDict['reference_sequence'] = reference_sequence
    entryDict['referenceCopyNumber'] = []
    for i in range(len(assembly_paths_list)):
        number_of_copies = find_number_of_copies_for_sequence(parser, assembly_paths_list[i], pdbName,
                                                              reference_sequence)
        if number_of_copies is None:  # not valid assembly
            entryDict['assemblies'].remove(i + 1)
            continue
        entryDict['referenceCopyNumber'].append(number_of_copies)
    return entryDict


def create_list_of_entry_dicts(parser, valid_UBD_candidates, valid_PDB_names, assemblyPathsLists):
    try:
        assert len(valid_PDB_names) == len(assemblyPathsLists)
        assert len(valid_PDB_names) == len(valid_UBD_candidates)
    except:

        print(len(valid_PDB_names))
        print(len(assemblyPathsLists))
        print(len(valid_UBD_candidates))
        assert (1 == 0)

    entryDictList = []
    for i in range(len(valid_PDB_names)):
        entryDictList.append(
            create_entry_dict(parser, i, valid_PDB_names[i], assemblyPathsLists[i], valid_UBD_candidates[i]))
    return entryDictList


def pickle_list_of_entry_dicts(list_of_entry_dicts, lists_size):
    """
    :param list_of_entry_dicts:
    :return:
    """
    addString = "_6"
    pickle_dir_path = os.path.join(paths.entry_dicts_path, f'PickleItems{str(lists_size)}{addString}')
    for i in range(len(list_of_entry_dicts) // lists_size):
        with open(os.path.join(pickle_dir_path, f'listOfEntryDicts{str(i)}'), "wb") as f:
            # pickle the list to the file
            pickle.dump(list_of_entry_dicts[i * lists_size:i * lists_size + lists_size], f)

    with open(pickle_dir_path + "\\listOfEntryDicts" + str(len(list_of_entry_dicts) // lists_size), "wb") as f:
        # pickle the list to the file
        pickle.dump(list_of_entry_dicts[(len(list_of_entry_dicts) // lists_size) * lists_size:], f)


def pickle_list_of_entry_dicts_in_one(i, list_of_entry_dicts):
    """
    :param list_of_entry_dicts:
    :return:
    """
    addString = str(i)
    pickleDirPath = 'C:\\Users\\omriy\\pythonProject\\ubiq_project\\pickle150New'
    with open(pickleDirPath + "\\listOfEntryDicts" + addString, "wb") as f:
        # pickle the list to the file
        pickle.dump(list_of_entry_dicts, f)
        print(len(list_of_entry_dicts))


def choose_assembly(entry_assembly_dict, probabilities, ambiguous_file, not_valid_file):
    """
    :param entry_assembly_dict: entryDict of pdb
    :param probabilities: Queen algorithm predictions
    :return: The path of the most likelihood assembly
    """
    # reference_copy_number_string = ' '.join(map(str, entry_assembly_dict['referenceCopyNumber']))
    # assemblies_string = ' '.join(map(str, entry_assembly_dict['assemblies']))
    # probabilities_string = ' '.join(["(" + str(INV_MAP[i]) + "," + str(probabilities[i]) + ")" for i in range(12)])
    # if len(entry_assembly_dict['referenceCopyNumber']) == 0:  # there werent any valid assemblies
    #     print("not valid")
    #     print(entry_assembly_dict['entry'])
    #     not_valid_file.write(
    #         entry_assembly_dict['entry'] + ": There werent any valid assemblies.\n longestNonUbiqFromAsymetric: " +
    #         entry_assembly_dict[
    #             'reference_sequence'] + "\n ,reference CopyNumber is : " + reference_copy_number_string + "\n")
    #     return None

    # predictions = []
    # for val in entry_assembly_dict['referenceCopyNumber']:
    #     if val in OPPOSITE_MAP.keys():
    #         predictions.append(probabilities[OPPOSITE_MAP[val]])
    # if len(predictions) == 0:  # there werent any valid assemblies
    #     not_valid_file.write(
    #         entry_assembly_dict['entry'] + ": There werent any valid assemblies.\n longestNonUbiqFromAsymetric: " +
    #         entry_assembly_dict['reference_sequence'] + "\n ,reference CopyNumber is : " + entry_assembly_dict[
    #             'referenceCopyNumber'] + "\n")
    #     return None
    # if len(predictions) > 1:
    #     ambiguous_file.write(entry_assembly_dict[
    #                              'entry'] + ": valid assembly numbers are: " + assemblies_string + "\n respective copyNumbers are: "
    #                          + reference_copy_number_string + " and respective propbabilities are :" + ' '.join(
    #         map(str, predictions)) + "\n the total probabilities are: " + probabilities_string
    #                          + "\n")
    
    # sorted_indexes = sorted(range(len(predictions)), key=lambda k: predictions[k], reverse=True)
    # for i in range(len(sorted_indexes)):
    #     assemblyPath = entry_assembly_dict['assemblyPathsList'][sorted_indexes[i]]
    #     structure = parser.get_structure(entry_assembly_dict['entry'], assemblyPath)
    #     candidate = UBD_candidate(structure)
    #     if len(candidate.models) > 0:
    #         return assemblyPath

    # not_valid_file.write("didnt find ubiq chain in any of the assemblies for " + entry_assembly_dict['entry'] +
    #                      "\n longestNonUbiqFromAsymetric: " + entry_assembly_dict['reference_sequence'])
    # return None  # no valid assembly found
    reference_copy_number_string = ' '.join(map(str, entry_assembly_dict['referenceCopyNumber']))
    assemblies_string = ' '.join(map(str, entry_assembly_dict['assemblies']))
    probabilities_string = ' '.join(["(" + str(INV_MAP[i]) + "," + str(probabilities[i]) + ")" for i in range(12)])
    if len(entry_assembly_dict['referenceCopyNumber']) == 0:  # there werent any valid assemblies
        print("not valid")
        print(entry_assembly_dict['entry'])
        not_valid_file.write(
            entry_assembly_dict['entry'] + ": There werent any valid assemblies.\n longestNonUbiqFromAsymetric: " +
            entry_assembly_dict[
                'reference_sequence'] + "\n ,reference CopyNumber is : " + reference_copy_number_string + "\n")
        return None

    predictions = []
    for val in entry_assembly_dict['referenceCopyNumber']:
        if val in OPPOSITE_MAP.keys():
            predictions.append(probabilities[OPPOSITE_MAP[val]])
    if len(predictions) == 0:  # there werent any valid assemblies
        not_valid_file.write(
            entry_assembly_dict['entry'] + ": There werent any valid assemblies.\n longestNonUbiqFromAsymetric: " +
            entry_assembly_dict['reference_sequence'] + "\n ,reference CopyNumber is : " + entry_assembly_dict[
                'referenceCopyNumber'] + "\n")
        return None
    if len(predictions) > 1:
        ambiguous_file.write(entry_assembly_dict[
                                 'entry'] + ": valid assembly numbers are: " + assemblies_string + "\n respective copyNumbers are: "
                             + reference_copy_number_string + " and respective propbabilities are :" + ' '.join(
            map(str, predictions)) + "\n the total probabilities are: " + probabilities_string
                             + "\n")

    maxPrediction = max(predictions)
    maxIndex = predictions.index(maxPrediction)
    count = entry_assembly_dict['referenceCopyNumber'][maxIndex]
    assemblyNum = entry_assembly_dict['assemblies'][maxIndex]
    assemblyPath = entry_assembly_dict['assemblyPathsList'][maxIndex]
    return assemblyPath



def choose_assemblies(listOfEntryDicts, listOfProbabillities, ambiguousFile, notValidFile):
    chosenAssembliesList = []
    for i in range(len(listOfEntryDicts)):
        assemblyPath = choose_assembly(listOfEntryDicts[i], listOfProbabillities[i], ambiguousFile, notValidFile)
        if assemblyPath is None:
            continue
        chosenAssembliesList.append(assemblyPath)
    return chosenAssembliesList


def from_pickle_to_choose_assemblies(entry_dicts_with_probabilities_path,assemblies_dir):
    entry_dicts_with_probabilities = load_as_pickle(entry_dicts_with_probabilities_path)
    all_probabilities = [entry['probabilities'] for entry in entry_dicts_with_probabilities]
    not_valid_file = open(os.path.join(assemblies_dir, "notValidAssembliesFileNew.txt"), "w")
    ambiguous_file = open(os.path.join(assemblies_dir, 'ambiguousFileNew.txt'), "w")
    chosen_assemblies = choose_assemblies(entry_dicts_with_probabilities, all_probabilities, ambiguous_file,
                                          not_valid_file)
    save_as_pickle(chosen_assemblies, os.path.join(assemblies_dir, 'chosen_assemblies.pkl'))
    not_valid_file.close()
    ambiguous_file.close()
    print(chosen_assemblies)
    return chosen_assemblies


def atom_dist(atom1, atom2):
    """
    :param atom1: atom object
    :param atom2: atom object
    :return: the euclidian distance between the atoms
    """

    vector1 = atom1.get_vector()
    vector2 = atom2.get_vector()
    temp = vector1 - vector2  # subtracting vector
    sum_sq = np.dot(temp, temp)  # sum of the squares
    return np.sqrt(sum_sq)


def get_label_for_aa(aa, ubiq_atoms, threshold, diameter=50, diameter_aa=8.):
    """
    :param aa: amino acid object
    :param ubiq_atoms: the ubiquitin atoms
    :return: 1 if there exists an atom that is within 4 Angstrom to a ubiquitin atom else 0
    """
    for atom in aa.get_atoms():
        dists = atom_dists(atom, ubiq_atoms)
        if dists.min() < threshold:
            return 1
        elif dists.max() > diameter + diameter_aa + threshold:
            return 0
    return 0


def get_labels_for_amino_acids(amino_acids, ubiq_atoms, amino_acids_labels_list, diameter):
    """
    :param amino_acids: list of chain's amino acid
    :param ubiq_atoms: ubiquitin atoms
    :param amino_acids_labels_list: list of the amino acids labels to be updated
    :return: True iff there is a connection between the chain and the ubiquitin(2 atoms within the threshold distance)
    """
    threshold = 4
    chain_ubiq_connection = False
    for i in range(len(amino_acids)):
        if get_label_for_aa(amino_acids[i], ubiq_atoms, threshold, diameter):
            chain_ubiq_connection = True
            amino_acids_labels_list[i] = '2'
    return chain_ubiq_connection


def fill_atrributes_amino_acids(amino_acids, chain_id, chain_attributes_matrix, amino_acids_labels_list):
    """
    :param amino_acids: list of chain's amino acid
    :param chain_id: The chain's id
    :param amino_acids_labels_list:
    :param chain_attributes_matrix:
    The function updates candidateAttributesMatrix such that candidateAttributesMatrix[j] = (chain_id, aa_id , aa_type, aa label)
    """

    for j in range(len(amino_acids)):
        chain_attributes_matrix[j][0] = chain_id
        chain_attributes_matrix[j][1] = str(amino_acids[j].get_id()[1])  # set amino acid id
        chain_attributes_matrix[j][2] = THREE_LETTERS_TO_SINGLE_AA_DICT[
            str(amino_acids[j].get_resname())]  # set amino acid type
        chain_attributes_matrix[j][3] = str(amino_acids_labels_list[j])


def check_connected_atoms_util(atomsA, atomsB, n, threshold):
    """
    :param atomsA: list of chain's atoms
    :param atomsB: list of chain's atoms
    :param n: number of atoms to check for chain A
    :param threshold: maximum distance to check between the atoms
    :return: True iff there are at least n pair of atoms (atomA,atomB) within threshold distance from eachother
    """
    cntPairs = 0
    for i in range(len(atomsA)):
        if atom_dists(atomsA[i], atomsB).min() < threshold:
            cntPairs += 1
            if cntPairs >= n:
                return True
    return False


def check_connected_atoms(aminoAcidsA, aminoAcidsB, n, threshold):
    """
    :param aminoAcidsA: list of chain's amino acid
    :param aminoAcidsB: list of chain's amino acid
    :param n: number of atoms to check for each chain
    :param threshold: maximum distance to check between the atoms
    :return: True iff there are at least n atoms in aminoAcidsA there are within threshold distance from aminoAcidsB
    and there are at least n atoms in aminoAcidsB there are within threshold distance from aminoAcidsA
    """
    atomsA = get_atoms_of_amino_acids(aminoAcidsA)
    atomsB = get_atoms_of_amino_acids(aminoAcidsB)
    if check_connected_atoms_util(atomsA, atomsB, n, threshold):
        return True
    return False


def create_ASA_list(model):
    asaDict = model.sasa_results_dict
    non_ubiq_chains_amino_acid_lists = [aa_out_of_chain(model.non_ubiq_chains[i]) for i in
                                        range(len(model.non_ubiq_chains))]
    AsaList = [
        [asaDict[non_ubiq_chains_amino_acid_lists[i][j].get_segid] for j in
         range(len(non_ubiq_chains_amino_acid_lists[i]))] for i
        in
        range(len(model.non_ubiq_chains))]
    return AsaList


def create_amino_acid_labels(model):
    """
    :param model:
    :return: Tuple : (ubiq_neighbors , non_ubiq_neighbors, model_attributes_matrix)
    model_attributes_matrix[i][j] = model_attributes_matrix[i] = (chain_id, aa_id , aa_type, aa label)
    """
    ubiq_diameters = [calculate_diameter_from_chain(model.ubiq_chains[i]) for i in range(len(model.ubiq_chains))]
    ubiq_neighbors = [[0 for j in range(len(model.ubiq_chains))] for i in range(len(model.non_ubiq_chains))]
    ubiq_chains_amino_acid_lists = [aa_out_of_chain(model.ubiq_chains[i]) for i in range(len(model.ubiq_chains))]
    ubiq_chains_atoms_lists = [get_atoms_of_amino_acids(ubiq_chains_amino_acid_lists[i]) for i in
                               range(len(model.ubiq_chains))]
    non_ubiq_chains_amino_acid_lists = [aa_out_of_chain(model.non_ubiq_chains[i]) for i in
                                        range(len(model.non_ubiq_chains))]
    model_labels_matrix = [[0 for j in range(len(non_ubiq_chains_amino_acid_lists[i]))] for i in
                           range(len(model.non_ubiq_chains))]
    model_attributes_matrix = [[[None, None, None, None] for j in range(len(non_ubiq_chains_amino_acid_lists[i]))] for i
                               in
                               range(len(model.non_ubiq_chains))]

    # --------- for each amino acid in the non ubiquitin chain, fill label , type and id stored in  model_attributes_matrix---------
    # --------- fill ubiq_neighbors matrix: ubiqNeigbors[i][j] == True <-> There is a connection between the i's non ubiquitin chain and the j's ubiquitin chain ---------
    for i in range(len(model.non_ubiq_chains)):  # iterare over the non ubiquitin chains
        # for i in range(6,7):
        for j in range(len(model.ubiq_chains)):  # iterare over the ubiquitin chains
            if get_labels_for_amino_acids(non_ubiq_chains_amino_acid_lists[i], ubiq_chains_atoms_lists[j],
                                          model_labels_matrix[
                                              i],
                                          ubiq_diameters[j]):  # there is a connection between the non ubiquitin chain and the ubiquitin chain
                ubiq_neighbors[i][j] = 1
        chain_id = model.non_ubiq_chains[i].get_id()
        fill_atrributes_amino_acids(non_ubiq_chains_amino_acid_lists[i], chain_id, model_attributes_matrix[i],
                                    model_labels_matrix[i])

    # --------- fill non_ubiq_neighbors matrix: ubiqNeigbors[i][j] == True <-> There is a connection between the i's non ubiquitin chain and the j's non ubiquitin chain ---------
    non_ubiq_neighbors = [[0 for j in range(len(model.non_ubiq_chains))] for i in range(len(model.non_ubiq_chains))]
    threshold = 4
    number_of_connected_atoms = 10
    for i in range(len(model.non_ubiq_chains)):  # iterare over the non ubiquitin chains
        for j in range(i, len(model.non_ubiq_chains)):  # iterare over the non ubiquitin chains
            if check_connected_atoms(non_ubiq_chains_amino_acid_lists[i], non_ubiq_chains_amino_acid_lists[j],
                                     number_of_connected_atoms, threshold):
                non_ubiq_neighbors[i][j] = 1
                non_ubiq_neighbors[j][i] = 1

    return (ubiq_neighbors, non_ubiq_neighbors, model_attributes_matrix)


def compute_connected_components(two_dim_list):
    """
    :param two_dim_list: A two dimensional list
    :return: Tuple(numComponents , componentsLabels)
    """
    np_non_ubiq_neighbors = np.array(two_dim_list)
    return connected_components(csgraph=np_non_ubiq_neighbors, directed=False, return_labels=True)


def create_related_chainslist(numberOfComponents, labels):
    """
    :param numberOfComponents: number of component = x => 0<=label values<x
    :param labels: labels
    :return: RelatedChainslist: RelatedChainslist[i] = list of all the chain index's which has the label i
    """
    related_chains_lists = [[] for _ in range(numberOfComponents)]
    for i in range(len(labels)):
        related_chains_lists[labels[i]].append(i)
    return related_chains_lists


# def connectivityAlgorithm(ubiqNeighbors,nonUbiqNeighbors):
def connectivity_algorithm(B, A):
    """
    :param B: (ubiqNeighbors)connectivity matrix (ndarray) of ubiquitin and non-ubiquitin chain in some candidate (dim: numberOf(non-ubiqChains) X numberOf(ubiqchains))
    :param A: (nonUbiqNeighbors)connectivity matrix (ndarray) of non-ubiquitin chains in some candidate (dim: numberOf(non-ubiqChains) X numberOf(non-ubiqChains))
    :return:
    """
    subset = B.sum(1)  # sum of rows (for each non-ubiq chain,number of connections to ubiquitin molecules)
    subset = subset > 0
    connection_index_list = subset.nonzero()

    A_S = A[subset, :][:, subset]  # sub-graph of non-ubiquitin graph of the chains interacting directly with ubiquitin
    B_S = B[subset, :]  # The corresponding connectivities
    C_S = np.dot(B_S, np.transpose(
        B_S))  # Connectivity matrix of non-ubiquitin chains interacting with at least one same ubiquitin chain
    D_S = np.multiply(A_S,
                      C_S)  # D_S[i][j] == 1 <-> chains i and j in direct contact and interact with at least one same ubiquitin chain
    num_components, components_labels = connected_components(csgraph=D_S, directed=False, return_labels=True)
    return num_components, components_labels, connection_index_list[0]


def create_imer_files(dirName):
    """
    :return: list of 24 files
    """
    return [open(os.path.join(dirName, f"Checkchains_{i}_mer.txt"), "w") for i in range(1, 25)]


def create_imer_asa_files(asa_dir_name):
    return [open(os.path.join(asa_dir_name, f"Checkchains_asa_{i}_mer.txt"), "w") for i in range(1, 25)]


def write_imer_to_file(file, model_attributes_matrix, ith_component_indexes_converted,
                       receptor_header):
    """
    :param model_attributes_matrix[i] = a list of the chain's amino acid attributes in the following format -(chain_id, aa_id , aa_type, aa label)
    :param ith_component_indexes_converted- list of model's chain's indexes
    """
    print(file.name)
    lines = []
    lines.append(">" + receptor_header)
    for i in ith_component_indexes_converted:
        for aminoAcidAttributes in model_attributes_matrix[i]:
            lines.append(" ".join(aminoAcidAttributes))
    stringToFile = "\n".join(lines)
    assert (file.write(stringToFile + "\n") > 0)


def write_asa_to_file(file, modelAttributesMatrix, ithComponentIndexesConverted,
                      receptorHeader,
                      ASAList):
    print(file.name)
    lines = []
    lines.append(">" + receptorHeader)
    for i in ithComponentIndexesConverted:
        for j in range(len(modelAttributesMatrix[i])):
            modelAttributesMatrix[i][j][3] = str(ASAList[i][j])
            lines.append(" ".join(modelAttributesMatrix[i][j]))
    stringToFile = "\n".join(lines)
    assert (file.write(stringToFile + "\n") > 0)


def update_labels_for_chains_util(imer_attributes_matrix, imer_amino_acids, non_binding_atoms, non_binding_diameter):
    threshold = 4
    for i in range(len(imer_amino_acids)):
        if get_label_for_aa(imer_amino_acids[i], non_binding_atoms, threshold, non_binding_diameter):
            if imer_attributes_matrix[i][3] == '0':  # doesn't bind ubiquitin
                imer_attributes_matrix[i][3] = '1'
            # else:
            elif imer_attributes_matrix[i][3] == '2':  # bind ubiquitin
                imer_attributes_matrix[i][3] = '3'


def update_labels_for_chain(imer_attributes_matrix, imer_amino_acids,
                            non_binding_atoms, non_binding_diameter):
    """
    :param imer_attributes_matrix: a list of the first chain's amino acid attributes in the following format -(chain_id, aa_id , aa_type, aa label)
    :param non_binding_attribute_matrix: a list of the second chain's amino acid attributes in the following format -(chain_id, aa_id , aa_type, aa label)
    """

    update_labels_for_chains_util(imer_attributes_matrix, imer_amino_acids, non_binding_atoms, non_binding_diameter)


def update_imers_labels(model_attributes_matrix, ith_component_indexes_converted, model, non_ubiq_diameters):
    """
    :param modelAttributesMatrix[i]: a list of the chain's amino acid attributes in the following format -(chain_id, aa_id , aa_type, aa label)
    :param ith_component_indexes_converted: list of model's chain's indexes
    """
    aminoAcidsLists = [aa_out_of_chain(model.non_ubiq_chains[index]) for index in range(len(model.non_ubiq_chains))]
    atomsLists = [get_atoms_of_amino_acids(aminoAcids) for aminoAcids in aminoAcidsLists]
    for i in ith_component_indexes_converted:
        for j in range(len(model.non_ubiq_chains)):
            if j not in ith_component_indexes_converted:  # one is binding non ubiquitin and one is a non-binding-non-ubiquitin
                update_labels_for_chain(model_attributes_matrix[i]
                                        , aminoAcidsLists[i],
                                        atomsLists[j],
                                        non_ubiq_diameters[j])


def convert_ubiq_binding_indexes_list(binding_indexes_list, ubiq_corresponding_list):
    l = []
    for i in binding_indexes_list:
        try:
            if ubiq_corresponding_list[i] is not None:
                l.append(ubiq_corresponding_list[i])
        except:
            print(i)
            print(ubiq_corresponding_list[i])
            assert (False)

    return l


def create_receptor_summary_util(model, ub_index, non_ub_index, bound_residue_set, non_ubiq_diameter):
    """
    :param bound_residue_set:
    :param model:
    :param ub_index:
    :param non_ub_index:
    :return: a list of the ubiquitin amino acid that bind to the non ubiquitin chain.
    """
    ubAminoAcids = aa_out_of_chain(model.ubiq_chains[ub_index])
    non_ub_amino_acids = aa_out_of_chain(model.non_ubiq_chains[non_ub_index])
    non_ub_atoms = get_atoms_of_amino_acids(non_ub_amino_acids)
    threshold = 4
    for i in range(len(ubAminoAcids)):
        if get_label_for_aa(ubAminoAcids[i], non_ub_atoms, threshold, non_ubiq_diameter):
            bound_residue_set.add(i)


def create_receptor_header(candidate, model, ith_component_indexes_converted):
    """
    :param candidate:
    :param model :
    :param ith_component_indexes_converted: indexes of the Imer's non ubiq chains
    """
    receptorHeader = str(candidate.structure.get_id()).lower() + "_" + '+'.join(
        [str(model.id) + "-" + model.non_ubiq_chains[i].get_id() for i in ith_component_indexes_converted])
    return receptorHeader


def create_receptor_summary(candidate, model, ubiq_neighbors, ith_component_indexes_converted, ubiq_corresponding_lists,
                            non_ubiq_diameters):
    """
    :param candidate:
    :param model:
    :param ubiq_neighbors: connectivity matrix (ndarray) of ubiquitin and non-ubiquitin chain in some candidate (dim: numberOf(non-ubiqChains) X numberOf(ubiqchains))
    :param ith_component_indexes_converted: non-ubiquitin chain indexes of the Receptor
    :return: a string of the following format (ReceptorHedear,NumUb,BoundResidueList)
    """
    bound_residue_sets = [set() for _ in range(len(model.ubiq_chains))]
    numUb = 0

    for j in range(len(model.ubiq_chains)):
        bind = False
        for index in ith_component_indexes_converted:
            if ubiq_neighbors[index][
                j] == 1:  # The j'th ubiquitin chain is binding with the index's non ubiquitin chain
                bind = True
                create_receptor_summary_util(model, j, index, bound_residue_sets[j], non_ubiq_diameters[index])
        if bind:
            numUb += 1
    boundResidueLists = [list(bound_residue_sets[i]) for i in range(len(bound_residue_sets))]

    convertedResidueLists = [convert_ubiq_binding_indexes_list(boundResidueLists[i], ubiq_corresponding_lists[i]) for i
                             in
                             range(len(boundResidueLists))]
    boundResidueStrings = ["+".join(convertedResidueLists[i]) for i in range(len(convertedResidueLists))]
    boundResidueStringsFiltered = [s for s in boundResidueStrings if s != ""]
    return ("//".join(boundResidueStringsFiltered), numUb)


import pdb


def create_data_base(tuple, ubiq_residus_list,ImerFiles_path,ASA_path):
    chosen_assemblies, index = tuple[0], tuple[1]
    try:
        index_string = str(index)
        assemblies_names = [chosen_assemblies[i].split("/")[-2].lower() for i in range(len(chosen_assemblies))]
        structures = [parser.get_structure(assemblies_names[i], chosen_assemblies[i]) for i in
                      range(len(chosen_assemblies))]
        UBD_candidates = [UBD_candidate(structure) for structure in structures]
        dirName = os.path.join(ImerFiles_path, f"Batch{index_string}")
        asa_dir_name = os.path.join(ASA_path, f"asaBatch{index_string}")
        print("\n\n\n creating dirs")
        if not os.path.exists(dirName):
            os.makedirs(dirName)
        if not os.path.exists(asa_dir_name):
            os.makedirs(asa_dir_name)
        files_list = create_imer_files(dirName)  # filesList[i] = file containing i-mers if created else None
        asa_files_list = create_imer_asa_files(asa_dir_name)
        summary_lines = []
        summary_file = open(os.path.join(dirName, "summaryLog.txt"), "w")
        for candidate in UBD_candidates:
            print(candidate.structure.get_id().lower())

            # if candidate.structure.get_id().lower() != '2n13':
            #     continue
            for model in candidate.models:
                non_ubiq_diameters = [calculate_diameter_from_chain(NonUbiqChain) for NonUbiqChain in
                                      model.non_ubiq_chains]
                asa_list = create_ASA_list(model)
                ubiq_neighbors, non_ubiq_neighbors, model_attributes_matrix = create_amino_acid_labels(model)
                np_non_ubiq_neighbors = np.array(non_ubiq_neighbors)
                np_ubiquitin_neighbors = np.array(ubiq_neighbors)
                num_components, components_labels, connection_index_list = connectivity_algorithm(
                    np_ubiquitin_neighbors,
                    np_non_ubiq_neighbors)
                ubiq_corresponding_lists = [
                    get_corresponding_ubiq_residues(get_str_seq_of_chain(ubiqChain), ubiq_residus_list) for ubiqChain in
                    model.ubiq_chains]

                for i in range(num_components):
                    ith_component_indexes = (components_labels == i).nonzero()[0]
                    ith_component_indexes_converted = []
                    for val in ith_component_indexes:
                        x = connection_index_list[val]
                        ith_component_indexes_converted.append(x)

                    receptor_header = create_receptor_header(candidate, model, ith_component_indexes_converted)
                    update_imers_labels(model_attributes_matrix, ith_component_indexes_converted, model,
                                        non_ubiq_diameters)
                    write_imer_to_file(files_list[len(ith_component_indexes_converted) - 1],
                                       model_attributes_matrix, ith_component_indexes_converted,
                                       receptor_header)
                    write_asa_to_file(asa_files_list[len(ith_component_indexes_converted) - 1],
                                      model_attributes_matrix, ith_component_indexes_converted,
                                      receptor_header, asa_list)
                    ubiquitin_binding_patch, number_of_bound_ubiq = create_receptor_summary(candidate, model,
                                                                                            ubiq_neighbors,
                                                                                            ith_component_indexes_converted,
                                                                                            ubiq_corresponding_lists,
                                                                                            non_ubiq_diameters)
                    number_of_receptors = len(ith_component_indexes_converted)
                    summary_lines.append(
                        '$'.join([receptor_header, str(number_of_receptors), str(number_of_bound_ubiq),
                                  ubiquitin_binding_patch]))

        summary_string = "\n".join(summary_lines)
        assert (summary_file.write(summary_string) > 0)
        summary_file.close()
        for file in files_list:
            file.close()
        for file in asa_files_list:
            file.close()
    except Exception as e:
        log_file = open(os.path.join(ImerFiles_path, f"Batch{str(index)}_log"), "w")
        log_file.write(f"An error occurred: {str(e)}\n")
        log_file.write(traceback.format_exc())
        log_file.close()



def split_list(original_list, num_sublists):
    sublist_size = len(original_list) // num_sublists
    remainder = len(original_list) % num_sublists

    result = []
    index = 0

    for i in range(num_sublists):
        sublist_length = sublist_size + 1 if i < remainder else sublist_size
        sublist = original_list[index:index + sublist_length]
        result.append(sublist)
        index += sublist_length

    return result
