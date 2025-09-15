import os
from typing import List
import tensorflow as tf
from data_preparation.patch_to_score.v1.schema.base import PatchToScoreProteinChain
from utils import save_as_pickle, save_as_tensor


NEGATIVE_SOURCES = set(
    ['Yeast proteome', 'Human proteome', 'Ecoli proteome', 'Celegans proteome', 'Arabidopsis proteome'])  # TODO: save elsewhere


def create_labels_from_sources(protein_chains: List[PatchToScoreProteinChain], save_dir_path: str):
    sources = [chain.source for chain in protein_chains]
    save_as_pickle(sources, os.path.join(
        save_dir_path, 'sources.pkl'))
    labels = tf.convert_to_tensor(
        [0 if source in NEGATIVE_SOURCES else 1 for source in sources])
    save_as_tensor(labels, os.path.join(
        save_dir_path, 'labels.tf'))
    return labels
