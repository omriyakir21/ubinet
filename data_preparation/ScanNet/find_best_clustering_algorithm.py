import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import paths
from data_preparation.ScanNet.db_creation_scanNet_utils import load_as_pickle,save_as_pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from sklearn.cluster import SpectralClustering, AffinityPropagation
from community import community_louvain  # Louvain
from leidenalg import find_partition, ModularityVertexPartition  # Leiden
import igraph as ig
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import silhouette_score
import data_preparation.ScanNet.cath_utils as cath_utils
import find_best_clustering_algorithm_utils as utils
from data_preparation.ScanNet.create_tables_and_weights import create_table,add_model_num_to_dataset

if __name__ == '__main__':
    dataset_date = "8_9"
    seq_id = "0.95"
    ASA_THRESHOLD_VALUE = 0.2
    with_scanNet = True
    with_scanNet_addition = '_with_scanNet_' if with_scanNet else ''


    pssm_folder = os.path.join(paths.PSSM_path,f'PSSM_{dataset_date}', f'seq_id_{seq_id}_asaThreshold_{ASA_THRESHOLD_VALUE}')
    
    full_pssm_file_path = os.path.join(os.path.join(pssm_folder, f'propagatedPssmWithAsaFile_{seq_id}_asaThreshold_{ASA_THRESHOLD_VALUE}{with_scanNet_addition}.txt'))
    cath_df = cath_utils.make_cath_df_new(os.path.join(paths.cath_path, "cath_b.20230204.txt"))

    names_list, sizes_list, sequence_list, full_names_list, pdb_names_with_chains_lists = cath_utils.list_creation(
        full_pssm_file_path)

    
    structuresDicts = cath_utils.create_dictionaries(names_list, sizes_list, sequence_list, full_names_list,
                                                     pdb_names_with_chains_lists)
    inCath, notInCath, cnt = cath_utils.count_in_cath(cath_df, structuresDicts)
    cath_utils.find_chains_in_cath(cath_df, structuresDicts)
    cath_utils.add_classifications_for_dict(cath_df, structuresDicts, 4)


    matHomologous_path = os.path.join(paths.cath_intermediate_files_path, f'matHomologous_{dataset_date}{with_scanNet_addition}.pkl')
    matHomologous = load_as_pickle(matHomologous_path)
    print(matHomologous.shape)
    graphHomologous = csr_matrix(matHomologous) 
    clusters_folder = os.path.join(paths.cath_intermediate_files_path, f'clusters{with_scanNet_addition}')
    os.makedirs(clusters_folder, exist_ok=True)
    graph = nx.from_scipy_sparse_array(graphHomologous)
    pos = nx.spring_layout(graph)
    
    test_partition_folder = os.path.join(paths.cath_intermediate_files_path, f'test_partition{with_scanNet_addition}')
    os.makedirs(test_partition_folder, exist_ok=True)
    algorithm_names = ["louvain", "leiden", "spectral", "affinity", "connected"]
    results = {}
    for algorithm_name in algorithm_names:
        print(f'processing algorithm: {algorithm_name}')
        algorithm_partition_folder = os.path.join(test_partition_folder, algorithm_name)
        os.makedirs(algorithm_partition_folder, exist_ok=True)
        n_clusters, labels = utils.create_components_for_algorithm(algorithm_name,graph,graphHomologous)
        chainDict = cath_utils.components_to_chain_dict(n_clusters, labels,sizes_list,full_names_list)
        table_path = os.path.join(algorithm_partition_folder, f'table{with_scanNet_addition}.csv')
        if not os.path.exists(table_path):
            print(f'creating table for algorithm: {algorithm_name}')
            cath_utils.divide_pssm(chainDict, full_pssm_file_path,algorithm_partition_folder,with_scanNet_addition)
            for i in range(5):
                pssm_file = os.path.join(algorithm_partition_folder, f"PSSM{with_scanNet_addition}{str(i)}.txt")
                output_file = os.path.join(algorithm_partition_folder,f'labels_fold{i+1}{with_scanNet_addition}.txt')
                add_model_num_to_dataset(pssm_file, output_file)
            create_table(algorithm_partition_folder ,algorithm_partition_folder,"",with_scanNet_addition)
        cross_edges,cross_edges_percentage ,total_possible_edges= utils.all_cross_edges(structuresDicts,table_path,matHomologous)
        max_fold_weight_ratio  = utils.max_fold_weight_ratio(table_path)
        results[algorithm_name] = (cross_edges,cross_edges_percentage,max_fold_weight_ratio)
    utils.plot_clustering_comparison(results,test_partition_folder,total_possible_edges)
    utils.clustering_comparision_summary_csv(results, test_partition_folder)
    



