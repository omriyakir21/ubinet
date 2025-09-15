import numpy as np
from typing import List
from data_preparation.patch_to_score.v1.schema.base import PatchToScoreProteinChain, PatchToScorePatch


def get_patch_average_scannet_ubiquitin_score(patch: PatchToScorePatch) -> float:
    return np.average([amino_acid.scannet_ubiquitin_score for amino_acid in patch.amino_acids])


def keep_only_top_components(protein_chain: PatchToScoreProteinChain, max_number_of_components: int) -> PatchToScoreProteinChain:
    patches = protein_chain.patches
    patches = sorted(patches, key=get_patch_average_scannet_ubiquitin_score, reverse=True)
    patches = patches[:max_number_of_components]
    protein_chain.patches = patches
    return protein_chain


def keep_only_top_components_from_list(protein_chains: List[PatchToScoreProteinChain], max_number_of_components: int) -> List[PatchToScoreProteinChain]:
    return [keep_only_top_components(protein_chain, max_number_of_components) for protein_chain in protein_chains]
