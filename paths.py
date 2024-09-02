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
PSSM_path = os.path.join(datasets_scanNet_path, 'PSSM')
#       QUEEN
QUEEN_path = os.path.join(datasets_scanNet_path, 'QUEEN')
entry_dicts_path = os.path.join(QUEEN_path, 'entry_dicts')

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
ScanNet_Ub_module_path = os.path.join(models_path, 'ScanNet_Ub')
AF2_multimer_model_path = os.path.join(models_path, 'AF2_multimer')
#   structural_aligners
structural_aligners_path = os.path.join(models_path, 'structural_aligners')
#       dali_aligner
dali_aligner_dir_path = os.path.join(structural_aligners_path, 'dali_aligner')
#           DaliLite.v5 dir
DaliLite_v5_dir_path = os.path.join(dali_aligner_dir_path, 'DaliLite.v5')
#           dali_aligments
dali_aligments_path = os.path.join(dali_aligner_dir_path, 'dali_aligments')

#   patch_to_score
patch_to_score_model_path = os.path.join(models_path, 'patch_to_score')
#       scalers    
scalers_path = os.path.join(patch_to_score_model_path, 'scalers')
#       all_predictions_0304
patch_to_score_all_predictions_0304_models_path = os.path.join(patch_to_score_model_path, 'all_predictions_0304')
#           with_MSA_50_plddt
with_MSA_50_plddt_0304_models_dir = os.path.join(patch_to_score_all_predictions_0304_models_path, 'with_MSA_50_plddt')


# results
results_path = os.path.join(current_dir, 'results')
#   AF3_multimer
AF3_predictions_path = os.path.join(results_path, 'AF3_predictions')
#   ScanNet
ScanNet_results_path = os.path.join(results_path, 'ScanNet')
#   patch_to_score
patch_to_score_results_path = os.path.join(results_path, 'patch_to_score')
#       all_predictions_0304
patch_to_score_all_predictions_0304_results_path = os.path.join(patch_to_score_results_path, 'all_predictions_0304')
#           with_MSA_50_plddt
with_MSA_50_plddt_0304_results_dir = os.path.join(patch_to_score_all_predictions_0304_results_path, 'with_MSA_50_plddt')


# tmp
tmp_path = os.path.join(current_dir, 'tmp')

# mafft exec
mafft_exec_path = '/home/iscb/wolfson/omriyakir/anaconda3/envs/ubinet/bin/mafft'

# mmseqs exec
mmseqs_exec_path = '/home/iscb/wolfson/omriyakir/anaconda3/envs/ubinet/bin/mmseqs'