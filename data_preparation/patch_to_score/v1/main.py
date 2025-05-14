import os
from utils import load_as_pickle, save_as_pickle
from data_preparation.patch_to_score.v1.compute_global_values.main import main as compute_global_values
from data_preparation.patch_to_score.v1.create_protein_chains.main import main as create_protein_chains
from data_preparation.patch_to_score.v1.top_patches.main import keep_only_top_components_from_list
from data_preparation.patch_to_score.v1.create_labels.main import create_labels_from_sources
from data_preparation.patch_to_score.v1.scale.main import main as scale
from data_preparation.patch_to_score.v1.partition.main import partition


def main(all_predictions_path: str,
         save_dir_path: str,
         sources_path: str,
         with_pesto: bool,
         plddt_threshold: float,
         should_override: float,
         max_number_of_components: int,
         sequence_identity: float,
         coverage: float,
         folds_amount: int,
         path2mmseqs: str,
         path2mmseqstmp: str):

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

    print('----> creating labels')  # TODO: account for negatives that appear in the positive set
    labels = create_labels_from_sources(
        protein_chains, os.path.join(save_dir_path, 'for_training'))

    data_for_training_dir_path = os.path.join(save_dir_path, 'for_training')
    
    # TODO: handle elsewhere (probably in chains creation, if necessary)
    save_as_pickle([chain.source for chain in protein_chains],os.path.join(data_for_training_dir_path, 'sources.pkl'))
    save_as_pickle([chain.uniprot_name for chain in protein_chains], os.path.join(data_for_training_dir_path, 'uniprots.pkl'))
    # save_as_pickle(protein_paths,os.path.join(data_for_training_dir_path, 'protein_paths.pkl'))
    
    print('----> scaling')
    # TODO: handle override
    scale(os.path.join(save_dir_path, 'scalers'),
          data_for_training_dir_path,
          protein_chains,
          max_number_of_components)

    print('----> partition')
    # TODO: handle override
    # TODO: GPU
    partition(sequences=[chain.sequence for chain in protein_chains],
              sequence_identity=sequence_identity,
              coverage=coverage,
              folds_amount=folds_amount,
              save_dir_path=data_for_training_dir_path,
              path2mmseqs=path2mmseqs,
              path2mmseqstmp=path2mmseqstmp)
