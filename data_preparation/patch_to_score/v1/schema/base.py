from typing import List, Dict
from dataclasses import dataclass
import networkx as nx
from Bio.PDB.Residue import Residue


@dataclass
class PatchToScoreAminoAcid:
    def __init__(self, residue: Residue, plddt: float,
                 scannet_protein_score: float, scannet_ubiquitin_score: float,
                 pesto_protein_score: float, pesto_dna_rna_score: float,
                 pesto_lipid_score: float, pesto_ligand_score: float,
                 pesto_ion_score: float):
        self.residue = residue
        self.plddt = plddt
        self.scannet_protein_score = scannet_protein_score
        self.scannet_ubiquitin_score = scannet_ubiquitin_score
        self.pesto_protein_score = pesto_protein_score
        self.pesto_dna_rna_score = pesto_dna_rna_score
        self.pesto_lipid_score = pesto_lipid_score
        self.pesto_ligand_score = pesto_ligand_score
        self.pesto_ion_score = pesto_ion_score
        
        self.features = {  # this will be scaled later on
            'plddt': plddt,
            'scanNet_protein': scannet_protein_score,
            'scanNet_ubiq': scannet_ubiquitin_score,
            'pesto_protein': pesto_protein_score,
            'pesto_dna_rna': pesto_dna_rna_score,
            'pesto_lipid': pesto_lipid_score,
            'pesto_ligand': pesto_ligand_score,
            'pesto_ion': pesto_ion_score,
            'ca_coord': residue['CA'].coord.copy(), 
        }
    
    def get_scores_dict(self) -> Dict[str, float]:  # TODO: made for backwards compatibility
        return {
            'plddt': self.plddt,
            'scanNet_protein': self.scannet_protein_score,
            'scanNet_ubiq': self.scannet_ubiquitin_score,
            'pesto_protein': self.pesto_protein_score,
            'pesto_dna_rna': self.pesto_dna_rna_score,
            'pesto_lipid': self.pesto_lipid_score,
            'pesto_ligand': self.pesto_ligand_score,
            'pesto_ion': self.pesto_ion_score
        }


@dataclass
class PatchToScorePatch:
    def __init__(self, amino_acids: List[PatchToScoreAminoAcid], component_set: list):    
        self.amino_acids = amino_acids
        self.component_set = component_set
        self.size = len(self.amino_acids)
        
        # TODO: do we want this? we didn't have this before
        self.features = {  # this will be scaled later on
            'size': self.size,
        }
    
    def get_average_scores_dict(self) -> Dict[str, float]:  # TODO: made for backwards compatibility
        average_scores = {}
        for aa in self.amino_acids:
            scores = aa.get_scores_dict()
            for key, value in scores.items():
                if key not in average_scores:
                    average_scores[key] = 0
                average_scores[key] += value
        for key in average_scores:
            average_scores[key] /= self.size
        return average_scores


@dataclass
class PatchToScoreProteinChain:
    def __init__(self, uniprot_name: str, source: str, sequence: str,
                 amino_acids: List[PatchToScoreAminoAcid], label: bool,
                 graph: nx.Graph, patches: List[PatchToScorePatch]):
        self.uniprot_name = uniprot_name
        self.source = source
        self.sequence = sequence
        self.amino_acids = amino_acids
        self.label = label  # True if the protein is positive - binds with ubiquitin, False if it is negative
        self.graph = graph
        self.patches = patches
        
        self.number_of_amino_acids = len(self.amino_acids)
        self.number_of_patches = len(self.patches)
        
        self.features = {  # this will be scaled later on
            'number_of_patches': self.number_of_patches,
            'number_of_amino_acids': self.number_of_amino_acids
        }
