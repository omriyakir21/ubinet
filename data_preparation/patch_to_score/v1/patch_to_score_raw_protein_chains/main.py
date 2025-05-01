import os
from typing import List
from tqdm import tqdm
from Bio.PDB import PDBParser
from Bio.PDB.Structure import Structure
from Bio.PDB.Residue import Residue
import numpy as np
from utils import save_as_pickle, load_as_pickle
from data_preparation.patch_to_score.v1.create_protein_objects.utils import create_patches_dict, create_90_percentile
from data_preparation.ScanNet.db_creation_scanNet_utils import aa_out_of_chain, get_str_seq_of_chain
from data_preparation.patch_to_score.v1.schema.base import PatchToScoreAminoAcid, PatchToScoreRawProteinChain

NEGATIVE_SOURCES = set(['Yeast proteome', 'Human proteome',
                       'Ecoli proteome', 'Celegans proteome', 'Arabidopsis proteome'])
POSITIVE_SOURCES = set(['E1', 'E2', 'E3', 'ubiquitinBinding', 'DUB'])
parser = PDBParser()


def create_paths(*paths: List[str]) -> None:
    for path in paths:
        os.makedirs(path, exist_ok=True)


def get_chain_predictions(uniprot_name: str, all_predictions: dict, with_pesto: bool) -> dict:
    predictions_dict = dict()
    predictions_dict['scanNet_ubiquitin'] = all_predictions['dict_predictions_ubiquitin'][uniprot_name]
    predictions_dict['scanNet_protein'] = all_predictions['dict_predictions_interface'][uniprot_name]
    if with_pesto:
        predictions_dict['pesto_protein'] = all_predictions['pesto_protein'][uniprot_name]
        predictions_dict['pesto_dna_rna'] = all_predictions['pesto_dna_rna'][uniprot_name]
        predictions_dict['pesto_ion'] = all_predictions['pesto_ion'][uniprot_name]
        predictions_dict['pesto_ligand'] = all_predictions['pesto_ligand'][uniprot_name]
        predictions_dict['pesto_lipid'] = all_predictions['pesto_lipid'][uniprot_name]
    return predictions_dict


def get_source_dir_path(source: str, sources_path: str) -> str:
    if source in NEGATIVE_SOURCES:
        source_dir_path = os.path.join(f'{sources_path}/AFDB',
                                       source.split(" ")[0])  # the name of the AFDB dirs doesnt end with proteome thats the reason of the split
    else:
        source_dir_path = os.path.join(f'{sources_path}/GO',
                                       source)
    return source_dir_path


def get_structure_path(uniprot_name: str, source: str, sources_path: str) -> str:
    source_dir_path = get_source_dir_path(source, sources_path)
    structure_path = os.path.join(source_dir_path, uniprot_name + '.pdb')
    return structure_path


def get_structure(uniprot_name: str, source: str, sources_path: str) -> Structure:
    structure_path = get_structure_path(uniprot_name, source, sources_path)
    if not os.path.exists(structure_path):
        raise Exception("path does not exist")
    structure = parser.get_structure(uniprot_name, structure_path)
    return structure


def get_sequence(structure: Structure) -> str:
    model = structure.child_list[0]
    assert (len(model) == 1)
    for chain in model:
        seq = get_str_seq_of_chain(chain)
    return seq


def get_plddt_values(structure: Structure):
    model = structure.child_list[0]
    assert (len(model) == 1)
    for chain in model:
        residues = aa_out_of_chain(chain)
        return np.array([residues[i].child_list[0].bfactor for i in range(len(residues))])


def create_amino_acid(residue: Residue, plddt_values: np.array, index_in_chain: int,
                      chain_predictions: dict, with_pesto: bool) -> PatchToScoreAminoAcid:
    # TODO : better handling of with_pesto
    amino_acid = PatchToScoreAminoAcid(residue=residue,
                                       plddt=plddt_values[index_in_chain],
                                       scannet_protein_score=chain_predictions['scanNet_protein'][index_in_chain],
                                       scannet_ubiquitin_score=chain_predictions[
                                           'scanNet_ubiquitin'][index_in_chain],
                                       pesto_protein_score=chain_predictions['pesto_protein'][
                                           index_in_chain] if with_pesto else None,
                                       pesto_dna_rna_score=chain_predictions['pesto_dna_rna'][
                                           index_in_chain] if with_pesto else None,
                                       pesto_ion_score=chain_predictions['pesto_ion'][
                                           index_in_chain] if with_pesto else None,
                                       pesto_ligand_score=chain_predictions['pesto_ligand'][
                                           index_in_chain] if with_pesto else None,
                                       pesto_lipid_score=chain_predictions['pesto_lipid'][index_in_chain] if with_pesto else None)

    return amino_acid


def create_raw_protein_chain(uniprot_name: str,
                             all_predictions: dict,
                             with_pesto: bool,
                             sources_path: str) -> PatchToScoreRawProteinChain:
    chain_predictions: dict = get_chain_predictions(
        uniprot_name, all_predictions, with_pesto)
    source = all_predictions['dict_sources'][uniprot_name]
    structure = get_structure(uniprot_name, source, sources_path)
    amino_acids = aa_out_of_chain(structure)
    sequence = get_sequence(structure)
    plddt_values = get_plddt_values(structure)
    pts_amino_acids = []
    for i, residue in enumerate(amino_acids):
        amino_acid = create_amino_acid(
            residue, plddt_values, i, chain_predictions, with_pesto)
        pts_amino_acids.append(amino_acid)
    raw_protein_chain = PatchToScoreRawProteinChain(uniprot_name=uniprot_name,
                                                    source=source,
                                                    sequence=sequence,
                                                    amino_acids=pts_amino_acids,
                                                    label=(source in POSITIVE_SOURCES))
    return raw_protein_chain


def main(all_predictions_path: str,
         save_dir_path: str,
         sources_path: str,
         uniprot_names: List[str],
         with_pesto: bool) -> None:
    """
    Create protein objects.

   :param str all_predictions_path: path to a pickle file containing all predictions in the form of a dictionary
   :param str save_dir_path: path to a directory where the protein objects will be saved
   :param str sources_path: path to a directory where the source .pdb files are saved
   :param list uniprot_names: list of unitprot names to create protein objects for
   :param bool with_pesto: should use pesto predictions
   :return: None
   :rtype: None
    """
    create_paths(save_dir_path)
    all_predictions = load_as_pickle(all_predictions_path)

    for uniprot_name in tqdm(uniprot_names):
        chain_save_path = os.path.join(
            save_dir_path, uniprot_name + '.pkl')
        if not os.path.exists(chain_save_path):
            raw_protein_chain = create_raw_protein_chain(
                uniprot_name, all_predictions, with_pesto, sources_path)
            save_as_pickle(raw_protein_chain, chain_save_path)  # TODO: should save by full source path?
