from typing import List
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


@dataclass
class PatchToScorePatch:
    amino_acids: List[PatchToScoreAminoAcid]
    component_set: list

    @property
    def size(self) -> int:
        return len(self.amino_acids) if (self.amino_acids is not None) else 0


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
