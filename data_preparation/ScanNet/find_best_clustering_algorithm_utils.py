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
import csv
from collections import defaultdict
import pandas as pd


def reorder_labels_by_cluster_size(labels):
    """Reorder labels such that label 0 is for the biggest cluster, label 1 for the second biggest, and so on."""
    unique, counts = np.unique(labels, return_counts=True)
    sorted_indices = np.argsort(-counts)  # Sort in descending order
    new_labels = np.zeros_like(labels)
    for new_label, old_label in enumerate(unique[sorted_indices]):
        new_labels[labels == old_label] = new_label
    return new_labels

def plot_cluster_sizes(labels, title):
    unique, counts = np.unique(labels, return_counts=True)
    plt.figure(figsize=(8, 5))
    plt.bar(unique, counts, alpha=0.7)
    plt.xlabel("Cluster ID")
    plt.ylabel("Size")
    plt.title(title)
    plt.savefig(os.path.join(clusters_folder, f'{title}{with_scanNet_addition}.png'))
    plt.close()

def prevent_imbalanced_clusters(labels, max_ratio=0.3):
    """Reassigns clusters if one cluster is too large relative to others."""
    unique, counts = np.unique(labels, return_counts=True)
    total = sum(counts)
    max_size = total * max_ratio
    if max(counts) > max_size:
        print("Warning: Large cluster detected. Consider alternative clustering or additional constraints.")
    return labels
def louvain_clustering(graph):
    partition = community_louvain.best_partition(graph)
    labels = np.array([partition[i] for i in range(len(partition))])
    number_of_clusters = len(set(labels))
    return number_of_clusters,labels

def leiden_clustering(graph):
    g = ig.Graph.Adjacency((nx.to_numpy_array(graph) > 0).tolist())
    partition = find_partition(g, ModularityVertexPartition)
    labels = np.zeros(graph.number_of_nodes(), dtype=int)
    for i, cluster in enumerate(partition):
        for node in cluster:
            labels[node] = i
    number_of_clusters = len(set(labels))
    return number_of_clusters,labels

def spectral_clustering(graph, n_clusters=80):
    clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', assign_labels='kmeans')
    labels = clustering.fit_predict(graph)
    return n_clusters,labels

def affinity_propagation(graph):
    graph_dense = graph.toarray()  # Convert sparse matrix to dense numpy array
    clustering = AffinityPropagation(affinity='precomputed')
    labels = clustering.fit_predict(graph_dense)
    number_of_clusters = len(set(labels))
    return number_of_clusters,labels

def connected_components_clustering(graph,directed=False):
    """Perform clustering using connected components."""
    n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
    return n_components, labels

def plot_graph_with_clusters(graph, labels, pos, title="Graph with Clusters", top_n=10):
    """Plot the graph with nodes colored according to their cluster labels, only coloring the top N clusters."""
    unique_labels, counts = np.unique(labels, return_counts=True)
    sorted_indices = np.argsort(-counts)  # Sort in descending order
    top_labels = unique_labels[sorted_indices[:top_n]]
    
    colors = plt.cm.rainbow(np.linspace(0, 1, top_n))  # Generate a color map for top N clusters

    plt.figure(figsize=(12, 8))
    for label, color in zip(top_labels, colors):
        nx.draw_networkx_nodes(graph, pos, nodelist=[node for node, lbl in enumerate(labels) if lbl == label],
                               node_color=[color], node_size=40, label=f"Cluster {label}")
    nx.draw_networkx_nodes(graph, pos, nodelist=[node for node, lbl in enumerate(labels) if lbl not in top_labels],
                           node_color='grey', node_size=20, label="Other Clusters")  # Smaller grey circles
    nx.draw_networkx_edges(graph, pos, alpha=0.5)
    plt.title(title)
    plt.legend()
    plt.savefig(os.path.join(clusters_folder, f'{title}_graph{with_scanNet_addition}.png'))
    plt.close()

def create_components_for_algorithm(algorithm_name,graph,graph_homologous):
    if algorithm_name == "louvain":
        return louvain_clustering(graph)
    elif algorithm_name == "leiden":
        return leiden_clustering(graph)
    elif algorithm_name == "spectral":
        return spectral_clustering(graph_homologous)
    elif algorithm_name == "affinity":
        return affinity_propagation(graph_homologous)
    elif algorithm_name == "connected":
        return connected_components_clustering(graph_homologous)
    else:
        raise ValueError(f"Unknown algorithm name: {algorithm_name}")


def calculate_max_fold_weight(file_path):
    fold_weights = defaultdict(float)
    total_weight = 0.0

    with open(file_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            fold = row['Set']
            weight = float(row['Sample weight'])
            fold_weights[fold] += weight
            total_weight += weight

    max_fold_weight = max(fold_weights.values())
    normalized_max_fold_weight = max_fold_weight / total_weight

    return normalized_max_fold_weight

def get_fold_dict(file_path):
    fold_dict = {}

    with open(file_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            pdb_id = row['PDB ID']
            fold_number = int(row['Set'].split()[-1]) - 1
            fold_dict[pdb_id] = fold_number

    return fold_dict

def get_index_dict(structuresDicts):
    index_dict = {}
    keys = list(structuresDicts.keys())
    for i in range(len(keys)):
        index_dict[keys[i]] = i

    return index_dict

def is_a_cross_edge(fold_dict,index_dict, id1, id2,matHomologous):
    index1 = index_dict[id1]
    index2 = index_dict[id2]
    return matHomologous[index1][index2] == 1 and fold_dict[id1] != fold_dict[id2]


def all_cross_edges(structuresDicts, table, mat_homologous):
    index_dict = get_index_dict(structuresDicts)
    fold_dict = get_fold_dict(table)
    print(f' len index_dict: {len(index_dict)}')
    print(f' len fold_dict: {len(fold_dict)}')  
    
    cross_edges = 0
    total_possible_edges = 0
    keys = list(fold_dict.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            id1 = keys[i]
            id2 = keys[j]
            total_possible_edges += 1
            if is_a_cross_edge(fold_dict, index_dict, id1, id2, mat_homologous):
                cross_edges += 1
    print(f'number of cross edges: {cross_edges}')
    
    if total_possible_edges > 0:
        cross_edges_percentage = (cross_edges / total_possible_edges) * 100
    else:
        cross_edges_percentage = 0
    
    print(f'percentage of cross edges: {cross_edges_percentage:.2f}%')

    return cross_edges, cross_edges_percentage, total_possible_edges

def max_fold_weight_ratio(table_path):
    table = pd.read_csv(table_path)
    total_weight = table['Sample weight'].sum()
    
    # Calculate the fold weights
    fold_weights = table.groupby('Set')['Sample weight'].sum()
    print(fold_weights)
    # Find the maximum fold weight
    max_fold_weight = fold_weights.max()
    
    # Calculate the ratio
    ratio = max_fold_weight / total_weight
    
    return ratio


def plot_clustering_comparison(results, dir_path, total_possible_edges):
    """
    Plots the comparison of clustering algorithms based on cross edges and max fold weight ratio.

    Parameters:
    results (dict): A dictionary where keys are algorithm names and values are tuples of (cross_edges, max_fold_weight_ratio).
    dir_path (str): The directory path where the plot image will be saved.
    total_possible_edges (int): The total number of possible edges.
    """
    # Plotting the results
    plt.figure(figsize=(10, 6))
    for algorithm_name, (_,cross_edges_percentage,max_fold_weight_ratio) in results.items():
        plt.scatter(cross_edges_percentage, max_fold_weight_ratio, label=algorithm_name)

    plt.xlabel('Cross Edges Percentage')
    plt.ylabel('Max Fold Weight Ratio')
    plt.title(f'Clustering Algorithm Comparison (Total Possible Edges: {total_possible_edges})')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(dir_path, 'max_fold_weight_ratio_vs_cross_edges.png'))
    plt.close()

def clustering_comparision_summary_csv(results, dir_path):
    """
    Saves the clustering comparison results to a CSV file.
    """
    with open(os.path.join(dir_path, 'clustering_comparison_summary.csv'), mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Algorithm', 'Cross Edges', 'Cross Edges Percentage', 'Max Fold Weight Ratio'])
        for algorithm_name, (cross_edges, cross_edges_percentage, max_fold_weight_ratio) in results.items():
            writer.writerow([algorithm_name, cross_edges, f"{cross_edges_percentage:.3f}", f"{max_fold_weight_ratio:.3f}"])
