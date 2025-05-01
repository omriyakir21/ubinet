import os
from data_preparation.patch_to_score.v1.compute_global_values.main import main as compute_global_values
from data_preparation.patch_to_score.v1.patch_to_score_raw_protein_chains import main as create_raw_protein_chains


def main(all_predictions_path: str, 
         save_dir_path: str,
         sources_path: str,
         with_pesto: bool = True):
    # ** inputs & outputs **
    # inputs:
    #   - all_predictions_path: path to a pickle file containing all predictions in the form of a dictionary
    #   - sources_path: path to a directory with all .pdb files
    #   - save_dir_path: path to a directory where the protein objects will be saved
    # outputs:
    #   - 'scaled_proteins' folders, with pickle files, each with a scaled protein, which includes everything needed
    #   - 'folds.pkl' file, which is a dictionary, where each fold maps to a protein object
    
    # ** steps **
    # 0. create base objects
    #       ** slurm over batches of proteins ** 
    #       parses single .pdb files, and predictions dictionary, to create protein objects.
    #       output here is a directory with protein object pickles.
    #       each protein has everything, besides pathces (set to None)
    # 1. compute global values
    #       ** slurm - single job, for loop ** 
    #       loops over all raw proteins, and computes global values over them all.
    #       for now this is the 90th percentile of the scannet ubiquitin binding score.
    #       saves to a directory. 
    # 2. create patches
    #       ** slurm over batches of proteins **
    #       parses single parsed proteins, and the global values, to create patches.
    #       creates protein objects with patches.
    #       saves to a directory.
    # 3. scale
    #       ** slurm over batches of proteins **
    #       loads the protein object and scales it
    #       saves it to a directory
    # 4. partition
    #       ** slurm - single job **
    #       loads all proteins together, and partitions them
    #       saves partitions to a directory
    
    pass
    
#     # TODO: split to slurm jobs
#     create_raw_protein_chains(all_predictions_path, 
#                               os.path.join(save_dir_path, 'objects'),
#                               sources_path,
#                               uniprot_names,
#                               with_pesto)
    
     # compute_global_values(all_predictions_path, os.path.join(save_dir_path, 'global_values'))
