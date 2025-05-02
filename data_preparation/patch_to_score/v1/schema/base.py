from typing import List, Union
from dataclasses import dataclass
import networkx as nx
from Bio.PDB.Residue import Residue
import numpy as np


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

    @property
    def calpha_coordinates(self) -> np.array:
        return [atom.coord for atom in self.atoms if atom.name == 'CA'][0]


@dataclass
class PatchToScorePatch:
    # a patch can be empty. In such case: amino_acids = None
    amino_acids: Union[None, List[PatchToScoreAminoAcid]]
    component_set: list

    @property
    def size(self) -> int:
        return len(self.amino_acids) if (self.amino_acids is not None) else 0

    @property
    def exists(self) -> bool:
        return self.size != 0


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
        return len([patch for patch in self.patches if patch.exists])

    @property
    def number_of_amino_acids(self) -> int:
        return len(self.amino_acids)
