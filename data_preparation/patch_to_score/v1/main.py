import os
from utils import load_as_pickle
from data_preparation.patch_to_score.v1.compute_global_values.main import main as compute_global_values
from data_preparation.patch_to_score.v1.create_protein_chains.main import main as create_protein_chains
from data_preparation.patch_to_score.v1.top_patches.main import keep_only_top_components


def main(all_predictions_path: str,
         save_dir_path: str,
         sources_path: str,
         with_pesto: bool,
         plddt_threshold: float,
         should_override: float,
         max_number_of_components: int):

    # TODO: slurm
    
    all_predictions = load_as_pickle(all_predictions_path)
    uniprot_names = list(all_predictions['dict_sources'].keys())  # TODO: (if needed) split to batches, then slurm

    percentile_90 = compute_global_values(
        all_predictions,
        os.path.join(save_dir_path, 'global_values'),
        should_override)

    protein_chains = create_protein_chains(all_predictions,
                                           os.path.join(
                                               save_dir_path, 'objects'),
                                           sources_path,
                                           uniprot_names,
                                           with_pesto,
                                           percentile_90,
                                           plddt_threshold,
                                           should_override)
    
    protein_chains = [keep_only_top_components(protein_chain, max_number_of_components) for protein_chain in protein_chains]

    # 3 & 4 are run together, in a slurm job, over all chains together
    # 3. scale
    #       loads the protein object and scales it
    #       saves it to a directory
    # 4. partition
    #       loads all proteins together, and partitions them
    #       saves partitions to a directory
