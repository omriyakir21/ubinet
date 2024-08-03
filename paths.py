import os

# List of directory and files paths
current_dir = os.path.dirname(os.path.abspath(__file__))

# data preparation paths
data_preparation_path = os.path.join(current_dir, 'data_preparation')
ScanNet_data_preparation_path = os.path.join(data_preparation_path, 'ScanNet')
# datasets paths
datasets_path = os.path.join(current_dir, 'datasets')
datasets_scanNet_path = os.path.join(datasets_path, 'ScanNet')
QUEEN_path = os.path.join(datasets_scanNet_path, 'QUEEN')
scanNet_sources_path = os.path.join(datasets_scanNet_path, 'sources')
blast_search_path = os.path.join(scanNet_sources_path, 'blast_search')
pdbs_path = os.path.join(scanNet_sources_path, 'pdbs')
assemblies_path = os.path.join(scanNet_sources_path, 'assemblies')
entry_dicts_path = os.path.join(QUEEN_path, 'entry_dicts')

# models
models_path = os.path.join(current_dir, 'models')
# results
results_path = os.path.join(current_dir, 'results')

