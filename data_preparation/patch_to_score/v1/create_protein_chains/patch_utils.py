from typing import List, Set
import networkx as nx
from data_preparation.patch_to_score.v1.schema.base import PatchToScoreAminoAcid, PatchToScorePatch


def create_patches(graph: nx.Graph, amino_acids: List[PatchToScoreAminoAcid]) -> List[PatchToScorePatch]:
    patches = []
    connected_components: List[Set[int]] = list(nx.connected_components(graph))
    for component_set in connected_components:
        patch_amino_acids = [amino_acids[i] for i in component_set]
        patch = PatchToScorePatch(patch_amino_acids, component_set)
        patches.append(patch)
    return patches
