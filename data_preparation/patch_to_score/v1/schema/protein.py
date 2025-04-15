import os

import numpy as np
import networkx as nx
from Bio.PDB import PDBParser

import paths
from data_preparation.ScanNet.db_creation_scanNet_utils import aa_out_of_chain, get_str_seq_of_chain


NEGATIVE_SOURCES = set(
    ['Yeast proteome', 'Human proteome', 'Ecoli proteome', 'Celegans proteome', 'Arabidopsis proteome'])
POSITIVE_SOURCES = set(['E1', 'E2', 'E3', 'ubiquitinBinding', 'DUB'])

parser = PDBParser()
DISTANCE_THRESHOLD = 10


def c_alpha_distance(atom1, atom2):
    vector1 = atom1.get_coord()
    vector2 = atom2.get_coord()
    distance = np.sqrt(((vector2[np.newaxis] - vector1[np.newaxis]) ** 2).sum(-1))
    return distance


class SizeDifferentiationException(Exception):
    def __init__(self, uniprotName):
        super().__init__("uniprotName: ", uniprotName, "\n")


class Protein:
    def __init__(self, uniprot_name, plddt_threshold, all_predictions, percentile_90, with_pesto):
        self.uniprot_name = uniprot_name
        self.percentile_90 = percentile_90
        self.predictions_dict = self.fill_predictions_dict(
            all_predictions, with_pesto)
        self.source = all_predictions['dict_sources'][uniprot_name]
        self.plddt_values = self.get_plddt_values()
        self.sequence = self.get_sequence()
        self.size = len(self.sequence)
        self.graph = nx.Graph()
        self.create_graph(plddt_threshold)
        self.connected_components_tuples = self.create_connected_components_tuples()

    def fill_predictions_dict(self, all_predictions, with_pesto):
        predictions_dict = {}
        predictions_dict['scanNet_ubiq'] = all_predictions['dict_predictions_ubiquitin'][self.uniprot_name]
        predictions_dict['scanNet_protein'] = all_predictions['dict_predictions_interface'][self.uniprot_name]
        if with_pesto:
            predictions_dict['pesto_protein'] = all_predictions['pesto_protein'][self.uniprot_name]
            predictions_dict['pesto_dna_rna'] = all_predictions['pesto_dna_rna'][self.uniprot_name]
            predictions_dict['pesto_ion'] = all_predictions['pesto_ion'][self.uniprot_name]
            predictions_dict['pesto_ligand'] = all_predictions['pesto_ligand'][self.uniprot_name]
            predictions_dict['pesto_lipid'] = all_predictions['pesto_lipid'][self.uniprot_name]
        return predictions_dict

    def get_residues(self):
        structure = self.get_structure()
        model = structure.child_list[0]
        assert (len(model) == 1)
        for chain in model:
            return aa_out_of_chain(chain)

    def get_structure(self, path=None):
        if path is not None:
            structurePath = path
        else:
            if self.source in NEGATIVE_SOURCES:
                structurePath = os.path.join(paths.AFDB_source_patch_to_score_path, self.source.split(" ")[0],
                                             self.uniprot_name + '.pdb')  # the name of the AFDB dirs doesnt end with proteome thats the reason of the split
            else:
                structurePath = os.path.join(paths.GO_source_patch_to_score_path, self.source,
                                             self.uniprot_name + '.pdb')
        print(structurePath)
        if not os.path.exists(structurePath):
            # print(f"path does not exist for : {self.uniprot_name}")
            # structurePath2 = all_predictions['dict_pdb_files'][self.uniprot_name]
            # convert_cif_to_pdb(structurePath2, structurePath)
            # print(f"created new path in : {structurePath}")
            raise Exception("path does not exist")
        structure = parser.get_structure(self.uniprot_name, structurePath)
        return structure

    def get_plddt_values(self):
        structure = self.get_structure()
        model = structure.child_list[0]
        assert (len(model) == 1)
        for chain in model:
            residues = aa_out_of_chain(chain)
            return np.array([residues[i].child_list[0].bfactor for i in range(len(residues))])

    def create_nodes_for_graph(self, residues, plddt_threshold):
        nodes = []
        if len(residues) != len(self.predictions_dict['scanNet_ubiq']):
            raise SizeDifferentiationException(self.uniprot_name)
        for i in range(len(residues)):
            plddtVal = residues[i].child_list[0].bfactor
            # print(f'plddt_val {plddtVal}')
            # print(f'threshold {plddt_threshold}')
            # print(f' self.predictions_dict[scanNet_ubiq][i] {self.predictions_dict["scanNet_ubiq"][i]}')
            # print(f'self.percentile_90 {self.percentile_90}')
            if plddtVal > plddt_threshold and self.predictions_dict['scanNet_ubiq'][i] > self.percentile_90:
                nodes.append(i)
        return nodes

    def create_edges_for_graph(self, residues, nodes):
        edges = []
        c_alpha_atoms = [residue["CA"] for residue in residues]
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                if c_alpha_distance(c_alpha_atoms[i], c_alpha_atoms[j]) < DISTANCE_THRESHOLD:
                    edges.append((nodes[i], nodes[j]))
        return edges

    def get_sequence(self):
        structure = self.get_structure()
        model = structure.child_list[0]
        assert (len(model) == 1)
        for chain in model:
            seq = get_str_seq_of_chain(chain)
        return seq

    def create_graph(self, plddt_threshold):
        structure = self.get_structure()
        model = structure.child_list[0]
        assert (len(model) == 1)
        for chain in model:
            residues = aa_out_of_chain(chain)
            nodes = self.create_nodes_for_graph(residues, plddt_threshold)
            valid_residues = [residues[i] for i in nodes]
            edges = self.create_edges_for_graph(valid_residues, nodes)
            self.graph.add_nodes_from(nodes)
            self.graph.add_edges_from(edges)

    def create_connected_components_tuples(self):
        tuples = []
        connected_components = list(nx.connected_components(self.graph))
        for component_set in connected_components:
            # average_ubiq, average_non_ubiq, average_plddt = self.calculate_average_predictions_for_component(
            # component_set)
            patch_dict = self.calculate_average_predictions_for_component(
                component_set)
            patch_size = len(component_set)
            tuples.append((patch_size, patch_dict, list(component_set)))
        return tuples

    def calculate_average_predictions_for_component(self, indexSet):
        patch_dict = {}
        indexes = list(indexSet)
        plddt_values = [self.plddt_values[index] for index in indexes]
        for key, val in self.predictions_dict.items():
            patch_dict[f'average_{key}'] = sum(
                [val[index] for index in indexes]) / len(indexes)
        patch_dict['average_plddt'] = sum(plddt_values) / len(plddt_values)
        return patch_dict
