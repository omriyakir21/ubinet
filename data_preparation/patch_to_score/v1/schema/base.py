from typing import List, Dict
from dataclasses import dataclass
import networkx as nx
from Bio.PDB.Residue import Residue


@dataclass
class PatchToScoreAminoAcid:
    residue: Residue
    plddt: float
    scannet_protein_score: float
    scannet_ubiquitin_score: float
    pesto_protein_score: float
    pesto_dna_rna_score: float
    pesto_lipid_score: float
    pesto_ligand_score: float
    pesto_ion_score: float
    
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
    amino_acids: List[PatchToScoreAminoAcid]
    component_set: list

    @property
    def size(self) -> int:
        return len(self.amino_acids) if (self.amino_acids is not None) else 0
    
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
    uniprot_name: str
    source: str
    sequence: str
    amino_acids: List[PatchToScoreAminoAcid]
    label: bool  # True if the protein is positive - binds with ubiquitin, False if it is negative
    graph: nx.Graph
    patches: List[PatchToScorePatch]

    @property
    def number_of_patches(self) -> int:
        return len(self.patches)

    @property
    def number_of_amino_acids(self) -> int:
        return len(self.amino_acids)
