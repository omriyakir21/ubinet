import os
import json
import random
import networkx as nx
import pandas as pd
import numpy as np
import cath_utils
import paths
import copy
from data_preparation.ScanNet.find_best_clustering_algorithm_utils import louvain_clustering,connected_components_clustering,spectral_clustering
from data_preparation.ScanNet.create_tables_and_weights import read_labels,add_model_num_to_dataset,calculate_weights
from data_preparation.ScanNet.db_creation_scanNet_utils import save_as_pickle, load_as_pickle
from data_preparation.ScanNet.LabelPropagationAlgorithm_utils import list_creation,filter_pssm_using_keys
from data_preparation.ScanNet.cath_utils import create_related_chainslist
# -----------------------------------------------------------------------------
# Helper to create a smaller debug PSSM from the full ScanNet source
# -----------------------------------------------------------------------------

def sample_pssm_file(input_pssm, output_pssm_debug, sample_n, seed=42):
    """
    Create a smaller sample of the PSSM file by randomly sampling chain blocks.

    Args:
      input_pssm (str): path to original PSSM file
      output_pssm_debug (str): path to write sampled PSSM
      sample_n (int): number of chain records to sample
      seed (int): random seed for reproducibility

    Returns:
      str: path to the debug PSSM file
    """
    records = []
    with open(input_pssm, 'r') as f:
        current = []
        for line in f:
            if line.startswith('>'):
                if current:
                    records.append(current)
                current = [line]
            else:
                current.append(line)
        if current:
            records.append(current)

    random.seed(seed)
    sampled = random.sample(records, sample_n) if sample_n < len(records) else records

    with open(output_pssm_debug, 'w') as f:
        for rec in sampled:
            f.writelines(rec)

    return output_pssm_debug

# -----------------------------------------------------------------------------
# Wrap existing PSSM -> Louvain clustering logic
# -----------------------------------------------------------------------------

def chain_to_cluster_dict(labels, full_names):
    chain_to_cluster = {}
    for i in range(labels.shape[0]):
        chain_to_cluster[full_names[i]] = labels[i]
    return chain_to_cluster
    

def cluster_chains(pssm_file,clustring_algorithm):
    names, sizes, seqs, full_names, pdb_with_chains = list_creation(pssm_file)
    structuresDicts = cath_utils.create_dictionaries(names, sizes, seqs, full_names, pdb_with_chains)
    cath_utils.find_chains_in_cath(cath_df, structuresDicts)
    cath_utils.add_classifications_for_dict(cath_df, structuresDicts, 4)
    
    mat = cath_utils.neighbor_mat_new(structuresDicts)
    if clustring_algorithm == "louvain":
        G = nx.from_numpy_array(mat)
        components, labels = louvain_clustering(G)
    elif clustring_algorithm == "connected":
        G = cath_utils.csr_matrix(mat)
        components, labels = connected_components_clustering(G)
    elif clustring_algorithm == "spectral":
        G = cath_utils.csr_matrix(mat)
        components, labels = spectral_clustering(G)
    else:
        raise ValueError(f"Unknown clustering algorithm: {clustring_algorithm}")
    chain_to_cluster = chain_to_cluster_dict(labels, full_names)
    return chain_to_cluster, structuresDicts

# -----------------------------------------------------------------------------
# Greedy weighted 5-fold assignment for ubiq clusters
# -----------------------------------------------------------------------------

def divide_clusters_by_weight(chain_to_cluster, weight_dict, n_folds=5):
    clusters = {}
    for chain, cid in chain_to_cluster.items():
        clusters.setdefault(cid, []).append(chain)

    cluster_weights = {
        cid: sum(weight_dict[c] for c in chains)
        for cid, chains in clusters.items()
    }

    folds = {i: [] for i in range(n_folds)}
    fold_weights = {i: 0.0 for i in range(n_folds)}

    for cid, cw in sorted(cluster_weights.items(), key=lambda x: x[1], reverse=True):
        target = min(fold_weights, key=fold_weights.get)
        folds[target].extend(clusters[cid])
        fold_weights[target] += cw

    return folds, fold_weights

def folds_joining(scan_folds, ubiq_folds, structures_scan, structures_ubiq, weight_dict):
    """
    Join scan and ubiq folds by similarity across iterations.
    On each iteration, compute a similarity matrix between remaining scan-ubiq folds
    (number of cross-edges where is_similiar_chains is True),
    pick the most similar pair, record it, and remove both from further consideration.
    Returns a list of tuples: (scan_fold_idx, ubiq_fold_idx, similarity_score).
    """
    
    matches = []
    cross_edged_weight_folds_dict = {}
    ubiq_fold_assigments = {i:[] for i in range(len(scan_folds))}


    # work on copies so we don't mutate inputs
    rem_scan = dict(scan_folds)
    rem_ubiq = dict(ubiq_folds)
    matches_matrix = np.zeros((len(scan_folds), len(ubiq_folds)))
    # build the similarity matrix
    for u_idx, u_chains in rem_ubiq.items():
        for s_idx, s_chains in rem_scan.items():
            for sc in s_chains:
                struct_s = structures_scan[sc]
                for uc in u_chains:
                    struct_u = structures_ubiq[uc]
                    if cath_utils.is_similiar_chains(struct_s, struct_u):
                        matches_matrix[s_idx, u_idx] += weight_dict[sc] * weight_dict[uc]

    for _ in range(len(scan_folds)):
        s_best, u_best = max(((scan_key, ubiq_key) for scan_key in rem_scan.keys() for ubiq_key in rem_ubiq.keys()),
            key=lambda pair: matches_matrix[pair[0], pair[1]])
        cross_edges_weight = np.sum(matches_matrix[np.arange(matches_matrix.shape[0]) != s_best, u_best])
        # remove the matched folds
        rem_scan.pop(s_best)
        rem_ubiq.pop(u_best)
        cross_edged_weight_folds_dict[s_best] = cross_edges_weight
        for chain in ubiq_folds[u_best]:
            ubiq_fold_assigments[s_best].append(chain)

    
    cross_edged_weight_folds = [cross_edged_weight_folds_dict[i] for i in range(len(scan_folds))]
    return ubiq_fold_assigments, cross_edged_weight_folds
    
def assign_ubiq_clusters(ubiq_chain_to_cluster, scan_folds, structures_scan, structures_ubiq, weight_dict,weight_bound_for_ubiq_fold):
    """
    Assign ubiq clusters to folds based on inter-source edges.
    Ties are broken using the fold with the lowest total sample weight.
    Clusters are processed from heaviest (by sample weight) to lightest.

    Args:
        ubiq_chain_to_cluster (dict): mapping of ubiq chains to cluster IDs
        scan_folds (dict[int, list[str]]): fold index -> list of scanNet chains
        structures_scan (dict): metadata for scanNet chains
        structures_ubiq (dict): metadata for ubiq chains
        weight_dict (dict): chain ID -> sample weight

    Returns:
        dict: ubiq chain -> fold assignment aggregated by fold
    """
    # Group chains by cluster for ubiq chains
    ubiq_clusters = {}
    for chain, cid in ubiq_chain_to_cluster.items():
        ubiq_clusters.setdefault(cid, []).append(chain)

    # Calculate total ubiq weight and set fill threshold (22% of total)
    total_ubq_weight = sum(weight_dict[c] for chains in ubiq_clusters.values() for c in chains)
    # fill_threshold = total_ubq_weight * 0.205
    fill_threshold = total_ubq_weight * weight_bound_for_ubiq_fold

    fold_ubq_weight = {f: 0.0 for f in scan_folds}
    # Sort ubiq clusters by total sample weight (descending)
    sorted_clusters = sorted(
        ubiq_clusters.items(),
        key=lambda item: sum(weight_dict[c] for c in item[1]),
        reverse=True
    )

    cross_edged_weight_folds = [0 for i in range(len(scan_folds))]
    print(f"Total ubiq clusters: {len(sorted_clusters)}")

    ubiq_chain_to_fold = {}
    for cid, chains in sorted_clusters:
        cluster_weight = sum(weight_dict[c] for c in chains)
        # Count similar chains between ubiq cluster and each scan fold
        counts = {f: 0 for f in scan_folds}
        for chain in chains:
            struct_u = structures_ubiq[chain]
            for fold, scan_chains in scan_folds.items():
                for scan_chain in scan_chains:
                    struct_s = structures_scan[scan_chain]
                    if cath_utils.is_similiar_chains(struct_s, struct_u):
                        counts[fold] += weight_dict[scan_chain] * weight_dict[chain]

        # Sort folds based on similarity counts (descending order)
        sorted_folds = sorted(counts.items(), key=lambda item: item[1], reverse=True)
        selected_fold = None
        # Check folds in order, skipping those if adding this cluster would exceed threshold
        for fold, _ in sorted_folds:
            if fold_ubq_weight[fold] + cluster_weight <= fill_threshold:
                selected_fold = fold
                break
        # If all candidate folds are "filled", assign to the best fold regardless
        if selected_fold is None:
            selected_fold = sorted_folds[0][0]

        # Assign cluster chains to chosen fold
        for chain in chains:
            ubiq_chain_to_fold[chain] = selected_fold
        fold_ubq_weight[selected_fold] += cluster_weight
        # Update fold scan weight for completeness
        cross_edged_weight_folds[selected_fold] += sum(counts[f] for f in counts if f != selected_fold)

    fold_assignments = {} 
    for chain, fold in ubiq_chain_to_fold.items():
        fold_assignments.setdefault(fold, []).append(chain)
    return fold_assignments,cross_edged_weight_folds


# -----------------------------------------------------------------------------
# Save a summary of partition sizes per fold
# -----------------------------------------------------------------------------


def save_partition_info(scan_folds, ubiq_folds,cross_edged_weight_folds,weight_dict, out_json):
    """
    Save partition summary to JSON file.
    Includes sizes of ScanNet and ubiq folds, and total sample weights.
    """
    partition_info = {
        'scan_folds': {f: len(chains) for f, chains in scan_folds.items()},
        'ubiq_folds': {f: len(chains) for f, chains in ubiq_folds.items()},
        'scan_weights': {f: sum(weight_dict[c] for c in chains) for f, chains in scan_folds.items()},
        'ubiq_weights': {f: sum(weight_dict[c] for c in chains) for f, chains in ubiq_folds.items()},
        'cross_edged_weight_folds': {f: cross_edged_weight_folds[i] for i,f in enumerate(scan_folds.keys())},
    }

    
    with open(out_json, 'w') as f:
        json.dump(partition_info, f, indent=4)
    
    print(f'Partition summary saved to {out_json}')

def create_weight_dict(scan_pssm:str,ubiq_pssm:str,helper_dir:str, debug:bool,weight_dict_path:str):
    """
    Create a weight dictionary mapping chains to their weights.
    This is used to ensure balanced partitioning of clusters.
    """
    #concat the PSSM filesmake sure that in between the two files there is a single new line
    combined_pssm = os.path.join(helper_dir, 'combined_pssms.txt')
    with open(scan_pssm, 'r') as f1, open(ubiq_pssm, 'r') as f2, open(combined_pssm, 'w') as out_f:
        for line in f1:
            out_f.write(line)
            last_line = line
        # if the last line of the first file does not end with a new line, add one
        if last_line and not last_line.endswith('\n'):
            out_f.write('\n')
        for line in f2:
            out_f.write(line)
    combined_with_model = os.path.join(helper_dir, 'combined_pssms_with_model_num.txt')
    add_model_num_to_dataset(combined_pssm, combined_with_model)
    list_origins, list_sequences, _, _ = read_labels(combined_with_model)
    all_weights_v0 = np.ones(len(list_sequences))
    all_weights_v1 = calculate_weights(list_sequences, resolutions=[100, 95, 90, 70])
    all_weights_v2 = calculate_weights(list_sequences, resolutions=[95])
    all_weights_v3 = calculate_weights(list_sequences, resolutions=[70])
    table = pd.DataFrame({
        'PDB ID': list_origins,
        'Length': [len(sequence) for sequence in list_sequences],
        'Sample weight': all_weights_v1,
        'Sample weight none': all_weights_v0,
        'Sample weight flat95': all_weights_v2,
        'Sample weight flat70': all_weights_v3,
    })
    output_path = os.path.join(helper_dir, f'weights{"_debug" if debug else ""}.csv')
    table.to_csv(output_path, index=False)
    print(f'Weights table created at {output_path}')

    # create weight dict with pdb_id as key and sample weight as value
    weight_dict = {}
    for i, pdb_id in enumerate(list_origins):
        weight_dict[pdb_id] = all_weights_v1[i]

    save_as_pickle(weight_dict, weight_dict_path)
    
if __name__ == '__main__':
    plan_dict = {
        'name': "v4",
        'seq_id': "0.95",
        'ASA_THRESHOLD_VALUE': 0.1,
        'weight_bound_for_ubiq_fold':0.205,
        'debug':False,
        'create_weight_table':False,
        'partition_ubiqs': 'scanNet',  # 'seperate' or 'scanNet'
        }
    os.makedirs(paths.cath_intermediate_files_path, exist_ok=True)
    cath_intermediate_files_path = os.path.join(paths.cath_intermediate_files_path, plan_dict['name'])
    os.makedirs(cath_intermediate_files_path, exist_ok=True)
    cath_intermidiate_helper_files_path = os.path.join(cath_intermediate_files_path,'helper_files')
    os.makedirs(cath_intermidiate_helper_files_path, exist_ok=True)
    PSSM_path = os.path.join(paths.PSSM_path, plan_dict['name'])
    PSSM_seq_id_folder = os.path.join(PSSM_path,f'seq_id_{plan_dict["seq_id"]}_asaThreshold_{plan_dict["ASA_THRESHOLD_VALUE"]}')
    debug_addition = "_debug" if plan_dict['debug'] else ""
    scan_pssm = os.path.join(PSSM_seq_id_folder, f'propagatedPssmFile_scanNet_non_ubiq_homologs{plan_dict["seq_id"]}_asaThreshold_{plan_dict["ASA_THRESHOLD_VALUE"]}{debug_addition}.txt')
    ubiq_pssm = os.path.join(PSSM_seq_id_folder, f'propagatedPssmFile_ubiqs_with_scanNet_homologs{plan_dict["seq_id"]}_asaThreshold_{plan_dict["ASA_THRESHOLD_VALUE"]}{debug_addition}.txt')
    
    cath_df = cath_utils.make_cath_df_new(os.path.join(paths.cath_path, "cath_b.20230204.txt"))
    seperate_addition = "_seperate" if plan_dict['partition_ubiqs'] == 'seperate' else ""
    bound_addition = f"_{str(plan_dict['weight_bound_for_ubiq_fold']).split('.')[1]}" if plan_dict['partition_ubiqs'] == 'scanNet' else ""
    print(f"partitioning with params: {plan_dict}", flush=True)

    weight_dict_path = os.path.join(cath_intermidiate_helper_files_path, f'chain_weights{"_debug" if plan_dict["debug"] else ""}.pkl')
    if plan_dict['create_weight_table']:
        create_weight_dict(scan_pssm, ubiq_pssm, cath_intermidiate_helper_files_path, plan_dict['debug'], 
                         weight_dict_path)

    weight_dict = load_as_pickle(weight_dict_path)

    # Cluster scanNet and split into weighted folds
    scan_map, structs_scan = cluster_chains(scan_pssm,"louvain")
    scan_folds, scan_weights = divide_clusters_by_weight(scan_map, weight_dict)
    
    ubiq_map, structs_ubiq = cluster_chains(ubiq_pssm,"louvain")

    if plan_dict['partition_ubiqs'] == 'seperate':
        ubiq_folds, ubiq_weights = divide_clusters_by_weight(ubiq_map, weight_dict)
        ubiq_folds,cross_edged_weight_folds = folds_joining(scan_folds, ubiq_folds, structs_scan, structs_ubiq, weight_dict)
    else:
        # Cluster & assign ScanNet (debug subset)
        ubiq_folds,cross_edged_weight_folds = assign_ubiq_clusters(ubiq_map, scan_folds, structs_scan, structs_ubiq,weight_dict,plan_dict['weight_bound_for_ubiq_fold'])
        # scan_assign = assign_scanNet_clusters(scan_map, ubiq_folds, structs_ubiq, structs_scan,weight_dict)
    # Save partition summary
    out_json = os.path.join(cath_intermidiate_helper_files_path, f'partition_summary{seperate_addition}{bound_addition}{debug_addition}.json')
    save_partition_info(scan_folds, ubiq_folds,cross_edged_weight_folds,weight_dict, out_json)

    print('Debug partition complete. Summary at', out_json)
    folds = {}
    for f in ubiq_folds.keys():
        folds[f] = copy.deepcopy(ubiq_folds[f])+ copy.deepcopy(scan_folds[f])

    propagated_pssm_file_path = os.path.join(PSSM_seq_id_folder, f'propagatedPssmFile_{plan_dict["seq_id"]}_asaThreshold_{plan_dict["ASA_THRESHOLD_VALUE"]}{debug_addition}.txt')
    for index,keys in folds.items():
        pssm_fold_path = os.path.join(PSSM_seq_id_folder, f'PSSM{seperate_addition}{bound_addition}_{index}{debug_addition}.txt')
        filter_pssm_using_keys(propagated_pssm_file_path, keys, pssm_fold_path)
