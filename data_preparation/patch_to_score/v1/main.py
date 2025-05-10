import os
from utils import load_as_pickle
from data_preparation.patch_to_score.v1.compute_global_values.main import main as compute_global_values
from data_preparation.patch_to_score.v1.create_protein_chains.main import main as create_protein_chains
from data_preparation.patch_to_score.v1.top_patches.main import keep_only_top_components_from_list
from data_preparation.patch_to_score.v1.create_labels.main import create_labels_from_sources
from data_preparation.patch_to_score.v1.scale.main import main as scale


def main(all_predictions_path: str,
         save_dir_path: str,
         sources_path: str,
         with_pesto: bool,
         plddt_threshold: float,
         should_override: float,
         max_number_of_components: int):

    # TODO: slurm

    print('----> loading all predictions')
    all_predictions = load_as_pickle(all_predictions_path)
    # TODO: (if needed) split to batches, then slurm
    uniprot_names = list(all_predictions['dict_sources'].keys())

    print('----> computing global values')
    percentile_90 = compute_global_values(
        all_predictions,
        os.path.join(save_dir_path, 'global_values'),
        should_override)

    print('----> creating protein chains')
    protein_chains = create_protein_chains(all_predictions,
                                           os.path.join(
                                               save_dir_path, 'objects'),
                                           sources_path,
                                           uniprot_names,
                                           with_pesto,
                                           percentile_90,
                                           plddt_threshold,
                                           should_override)

    print('----> keeping only top components')
    protein_chains = keep_only_top_components_from_list(
        protein_chains, max_number_of_components)

    print('----> creating labels')
    labels = create_labels_from_sources(
        protein_chains, os.path.join(save_dir_path, 'for_training'))

    print('----> scaling')
    # TODO: handle override
    scale(os.path.join(save_dir_path, 'scalers'),
          os.path.join(save_dir_path, 'for_training'),
          protein_chains,
          max_number_of_components)

    # 3 & 4 are run together, in a slurm job, over all chains together
    # 3. scale
    #       loads the protein object and scales it
    #       saves it to a directory
    # 4. partition
    #       loads all proteins together, and partitions them
    #       saves partitions to a directory
