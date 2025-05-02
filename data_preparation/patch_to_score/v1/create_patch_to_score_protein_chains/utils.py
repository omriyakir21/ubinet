import os
from typing import List, Tuple
from tqdm import tqdm
import numpy as np

from Bio.PDB import PDBParser
from Bio.PDB.Structure import Structure
from Bio.PDB.Residue import Residue
from Bio.PDB.Atom import Atom
import networkx as nx

from utils import save_as_pickle, load_as_pickle, create_paths
from data_preparation.patch_to_score.v1.create_protein_objects.utils import create_patches_dict
from data_preparation.ScanNet.db_creation_scanNet_utils import aa_out_of_chain, get_str_seq_of_chain
from data_preparation.patch_to_score.v1.schema.base import PatchToScorePatch


NEGATIVE_SOURCES = set(
    ['Yeast proteome', 'Human proteome', 'Ecoli proteome', 'Celegans proteome', 'Arabidopsis proteome'])
POSITIVE_SOURCES = set(['E1', 'E2', 'E3', 'ubiquitinBinding', 'DUB'])

parser = PDBParser()
DISTANCE_THRESHOLD = 10


class SizeDifferentiationException(Exception):
    def __init__(self, uniprotName):
        super().__init__("uniprotName: ", uniprotName, "\n")
        
        
def c_alpha_distance(atom1: Atom, atom2: Atom) -> float:
    vector1 = atom1.get_coord()
    vector2 = atom2.get_coord()
    distance = np.sqrt(((vector2[np.newaxis] - vector1[np.newaxis]) ** 2).sum(-1))
    return distance
        

def get_structure(path: str, uniprot_name: str) -> Structure:
    if not os.path.exists(path):
        raise Exception("path does not exist")
    structure = parser.get_structure(uniprot_name, path)
    return structure


def create_nodes_for_graph(uniprot_name: str, residues: List[Residue],
                           scannet_ubiquitin_scores: np.array, 
                           plddt_threshold: float, percentile_90: float) -> List[int]:
    nodes = []
    if len(residues) != len(scannet_ubiquitin_scores):
        raise SizeDifferentiationException(uniprot_name)
    for i in range(len(residues)):
        plddtVal = residues[i].child_list[0].bfactor
        if plddtVal > plddt_threshold and scannet_ubiquitin_scores[i] > percentile_90:
            nodes.append(i)
    return nodes

def create_edges_for_graph(residues: List[Residue], nodes: List[int]) -> List[Tuple[int, int]]:
    edges = []
    c_alpha_atoms = [residue["CA"] for residue in residues]
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if c_alpha_distance(c_alpha_atoms[i], c_alpha_atoms[j]) < DISTANCE_THRESHOLD:
                edges.append((nodes[i], nodes[j]))
    return edges


def create_graph(plddt_threshold: float) -> nx.Graph:
    graph = nx.Graph()
    structure = get_structure()
    model = structure.child_list[0]
    assert (len(model) == 1)
    for chain in model:
        residues = aa_out_of_chain(chain)
        nodes = create_nodes_for_graph(residues, plddt_threshold)
        valid_residues = [residues[i] for i in nodes]
        edges = create_edges_for_graph(valid_residues, nodes)
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)
    return graph


def create_patches() -> List[PatchToScorePatch]:
    pass
