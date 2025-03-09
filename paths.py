import os

# List of directory and files paths
current_dir = os.path.dirname(os.path.abspath(__file__))

# data preparation paths
data_preparation_path = os.path.join(current_dir, 'data_preparation')
    #   patch_to_score
patch_to_score_path = os.path.join(data_preparation_path, 'patch_to_score')
mmseqs_patch_to_score_path = os.path.join(patch_to_score_path, 'mmseqs')

    #   ScanNet
ScanNet_data_preparation_path = os.path.join(data_preparation_path, 'ScanNet')
mmseqs_scanNet_path = os.path.join(ScanNet_data_preparation_path, 'mmseqs')
mafft_path = os.path.join(ScanNet_data_preparation_path, 'mafft')
ASA_path = os.path.join(ScanNet_data_preparation_path, 'ASA')
ImerFiles_path = os.path.join(ScanNet_data_preparation_path, 'ImerFiles')
cath_intermediate_files_path = os.path.join(ScanNet_data_preparation_path, 'cath_intermediate_files')


# datasets paths
datasets_path = os.path.join(current_dir, 'datasets')
#   patch_to_score
patch_to_score_dataset_path = os.path.join(datasets_path, 'patch_to_score')
#       patches dict
patches_dicts_path = os.path.join(patch_to_score_dataset_path, 'patches_dicts')
#       data_for_training
patch_to_score_data_for_training_path = os.path.join(patch_to_score_dataset_path, 'data_for_training')
#       sources
patch_to_score_sources_path = os.path.join(patch_to_score_dataset_path, 'sources')
#           GO
GO_source_patch_to_score_path = os.path.join(patch_to_score_sources_path, 'GO')
#               ubiquitinBinding
ubiquitin_binding_path = os.path.join(GO_source_patch_to_score_path, 'ubiquitinBinding')
#               E1
E1_path = os.path.join(GO_source_patch_to_score_path, 'E1')
#               E2
E2_path = os.path.join(GO_source_patch_to_score_path, 'E2')
#               E3
E3_path = os.path.join(GO_source_patch_to_score_path, 'E3')
#               DUB
DUB_path = os.path.join(GO_source_patch_to_score_path, 'DUB')
#           AFDB
AFDB_source_patch_to_score_path = os.path.join(patch_to_score_sources_path, 'AFDB')
#   scanNet
datasets_scanNet_path = os.path.join(datasets_path, 'scanNet')
#       binding_chains_pdbs
binding_chains_pdbs_path = os.path.join(datasets_scanNet_path, 'binding_chains_pdbs')
#           with ubiqs
binding_chains_pdbs_with_ubiqs_path = os.path.join(binding_chains_pdbs_path, 'with_ubiqs')
#           without ubiqs
binding_chains_pdbs_without_ubiqs_path = os.path.join(binding_chains_pdbs_path, 'without_ubiqs')
#       AF2_augmentations
scanNet_AF2_augmentations_path = os.path.join(datasets_scanNet_path, 'AF2_augmentations')
#           original_pdbs_with_augmentations
original_pdbs_with_augmentations_path = os.path.join(scanNet_AF2_augmentations_path, 'original_pdbs_with_augmentations')
#           pdbs_with_augmentations_95_path
pdbs_with_augmentations_95_path = os.path.join(scanNet_AF2_augmentations_path, 'pdbs_with_augmentations_95')
#           pdbs_with_augmentations_90_path 
pdbs_with_augmentations_90_path = os.path.join(scanNet_AF2_augmentations_path, 'pdbs_with_augmentations_90')

PSSM_path = os.path.join(datasets_scanNet_path, 'PSSM')
#       QUEEN
QUEEN_path = os.path.join(datasets_scanNet_path, 'QUEEN')
entry_dicts_path = os.path.join(QUEEN_path, 'entry_dicts')
#       data_for_trainnig
scanNet_data_for_training_path = os.path.join(datasets_scanNet_path, 'data_for_training')

#       sources
scanNet_sources_path = os.path.join(datasets_scanNet_path, 'sources')
cath_path = os.path.join(scanNet_sources_path, 'cath')
blast_search_path = os.path.join(scanNet_sources_path, 'blast_search')
pdbs_path = os.path.join(scanNet_sources_path, 'pdbs')
assemblies_path = os.path.join(scanNet_sources_path, 'assemblies')
chosen_assemblies_path = os.path.join(scanNet_sources_path, 'chosen_assemblies')

# models
models_path = os.path.join(current_dir, 'models')
QUEEN_model_path = os.path.join(models_path, 'QUEEN')
#   scanNet
ScanNet_Ub_module_path = os.path.join(models_path, 'ScanNet_Ub')
#       scanNet datasets
ScanNet_Ub_datasets = os.path.join(ScanNet_Ub_module_path,'datasets')
#           scanNet PPBS datasets
ScanNet_Ub_PPBS_datasets = os.path.join(ScanNet_Ub_datasets,'PPBS')
AF2_multimer_model_path = os.path.join(models_path, 'AF2_multimer')
#   structural_aligners
structural_aligners_path = os.path.join(models_path, 'structural_aligners')
#       pdb_files 
pdb_files_structural_aligners_path = os.path.join(structural_aligners_path, 'pdb_files')
#       missing uniprots fasta files
missing_uniprots_fasta_files_path = os.path.join(structural_aligners_path, 'missing_uniprots_fasta_files')
#       files from colab fold
structural_alignment_files_from_colab_path = os.path.join(structural_aligners_path, 'files_from_colab_fold')
#       dali_aligner
dali_aligner_dir_path = os.path.join(structural_aligners_path, 'dali_aligner')
#           DaliLite.v5 dir
DaliLite_v5_dir_path = os.path.join(dali_aligner_dir_path, 'DaliLite.v5')
#           dali_aligments
dali_aligments_path = os.path.join(dali_aligner_dir_path, 'dali_aligments')
#       TM-align
TM_aligner_dir_path = os.path.join(structural_aligners_path, 'TM_align')
#           TM-align aligned pdbs
TM_aligner_aligned_pdbs_path = os.path.join(TM_aligner_dir_path, 'aligned_pdbs')
#           TM-align transformed pdb files
TM_aligner_transformed_pdbs_path = os.path.join(TM_aligner_dir_path, 'transformed_pdbs')  
#   patch_to_score
patch_to_score_model_path = os.path.join(models_path, 'patch_to_score')



# results
results_path = os.path.join(current_dir, 'results')
#   chainsaw
chainsaw_results_path = os.path.join(results_path, 'chainsaw')
#   AF3_multimer
AF3_predictions_path = os.path.join(results_path, 'AF3_predictions')
#   ScanNet
ScanNet_results_path = os.path.join(results_path, 'ScanNet')
#   patch_to_score
patch_to_score_results_path = os.path.join(results_path, 'patch_to_score')
#       all_predictions_0304
patch_to_score_all_predictions_results_path = os.path.join(patch_to_score_results_path, 'all_predictions')
#           with_MSA_50_plddt
with_MSA_50_plddt_results_dir = os.path.join(patch_to_score_all_predictions_results_path, 'with_MSA_50_plddt')


# tmp
tmp_path = os.path.join(current_dir, 'tmp')

# mafft exec
mafft_exec_path = '/home/iscb/wolfson/omriyakir/anaconda3/envs/ubinet/bin/mafft'

# mmseqs exec
mmseqs_exec_path = '/home/iscb/wolfson/omriyakir/anaconda3/envs/ubinet/bin/mmseqs'