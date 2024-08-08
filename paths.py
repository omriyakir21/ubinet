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

# models
models_path = os.path.join(current_dir, 'models')
QUEEN_model_path = os.path.join(models_path, 'QUEEN')
ScanNet_Ub_module_path = os.path.join(models_path, 'ScanNet_Ub')
# results
results_path = os.path.join(current_dir, 'results')
#   ScanNet
ScanNet_results_path = os.path.join(results_path, 'ScanNet')
patches_dicts_path = os.path.join(ScanNet_results_path, 'patches_dicts')
# tmp
tmp_path = os.path.join(current_dir, 'tmp')

# mafft exec
mafft_exec_path = '/home/iscb/wolfson/omriyakir/anaconda3/envs/ubinet/bin/mafft'

# mmseqs exec
mmseqs_exec_path = '/home/iscb/wolfson/omriyakir/anaconda3/envs/ubinet/bin/mmseqs'