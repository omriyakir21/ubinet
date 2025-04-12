import sys
import os
from typing import List
import pickle
from itertools import chain
from plistlib import load
import networkx as nx
import os
import numpy as np


class PatchToScoreAminoAcid:
    """
    An amino-acid, as a part of a patch.
    TODO: size (?)
    """
    def __init__(self, x, y, z, resname, plddt, scannet_ubiq, scannet_protein,
                 pesto_protein, pesto_dna_rna, pesto_ion, pesto_ligand, pesto_lipid):
        self.x = x
        self.y = y
        self.z = z
        self.scannet_ubiq = scannet_ubiq
        self.scannet_protein = scannet_protein
        self.pesto_protein = pesto_protein
        self.pesto_dna_rna = pesto_dna_rna
        self.pesto_ion = pesto_ion
        self.pesto_ligand = pesto_ligand
        self.pesto_lipid = pesto_lipid
        self.plddt = plddt
        self.resname = resname


class PatchToScorePatch:
    """ 
    A patch contains a set of amino acids.
    """
    def __init__(self, amino_acids: List[PatchToScoreAminoAcid]):
        self.amino_acids = amino_acids
        self.number_of_amino_acids = len(amino_acids)
        
        
class PatchToScoreProtein:
    def __init__(self, uniprot_id: str, sequence: str,
                 patches: List[PatchToScorePatch],
                 plddt_threshold: float, percentile_90: float, source: str):
        self.uniprot_id = uniprot_id
        self.patches = patches
        self.number_of_patches = len(patches)
        self.sequence = sequence  # out of patches data, otherwise, protein object only sees patches
        self.number_of_amino_acids = len(sequence)
        
        self.plddt_threshold = plddt_threshold
        self.percentile_90 = percentile_90
        self.source = source
