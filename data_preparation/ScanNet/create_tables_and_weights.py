import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import subprocess, shutil
import pandas as pd
import paths
import datetime

def add_model_num_to_dataset(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        current_section = None
        for line in infile:
            if line.startswith('>'):
                if line[1]=='|':
                    current_section = line.split('|')[2].split('_')[1].split('-')[0]
                current_section = line.split('_')[1].split('-')[0]
                outfile.write(line)
            else:
                outfile.write(f"{current_section} {line}")

# Usage
def read_labels(input_file, nmax=np.inf, label_type='int'):
    list_origins = []
    list_sequences = []
    list_labels = []
    list_resids = []

    with open(input_file, 'r') as f:
        count = 0
        for line in f:
            if (line[0] == '>'):
                if count == nmax:
                    break
                if count > 0:
                    list_origins.append(origin)
                    list_sequences.append(sequence)
                    list_labels.append(np.array(labels))
                    list_resids.append(np.array(resids))

                origin = line[1:-1]
                sequence = ''
                labels = []
                resids = []
                count += 1
            else:
                line_splitted = line[:-1].split(' ')
                resids.append(line_splitted[:-2])
                sequence += line_splitted[-2]
                if label_type == 'int':
                    labels.append(int(line_splitted[-1]))
                else:
                    labels.append(float(line_splitted[-1]))

    list_origins.append(origin)
    list_sequences.append(sequence)
    list_labels.append(np.array(labels))
    list_resids.append(np.array(resids))

    list_origins = np.array(list_origins, dtype=object)
    list_sequences = np.array(list_sequences, dtype=object)
    list_labels = np.array(list_labels, dtype=object)
    list_resids = np.array(list_resids, dtype=object)
    return list_origins, list_sequences, list_resids, list_labels


def cluster_sequences(list_sequences, seqid=1.0, coverage=0.8, covmode='0', path2mmseqstmp=paths.tmp_path,
                      path2mmseqs=paths.mmseqs_exec_path
                      ):

    rng = np.random.randint(0, high=int(1e6))
    tmp_input = os.path.join(path2mmseqstmp, 'tmp_input_file_%s.fasta' % rng)
    tmp_output = os.path.join(path2mmseqstmp, 'tmp_output_file_%s' % rng)

    with open(tmp_input, 'w') as f:
        for k, sequence in enumerate(list_sequences):
            f.write('>%s\n' % k)
            f.write('%s\n' % sequence)

    command = ('{mmseqs} easy-cluster {fasta} {result} {tmp} --min-seq-id %s -c %s --cov-mode %s' % (
        seqid, coverage, covmode)).format(mmseqs=path2mmseqs, fasta=tmp_input, result=tmp_output, tmp=path2mmseqstmp)
    subprocess.run(command.split(' '))

    with open(tmp_output + '_rep_seq.fasta', 'r') as f:
        representative_indices = [int(x[1:-1]) for x in f.readlines()[::2]]
    cluster_indices = np.zeros(len(list_sequences), dtype=int)
    table = pd.read_csv(tmp_output + '_cluster.tsv', sep='\t', header=None).to_numpy(dtype=int)
    for i, j in table:
        if i in representative_indices:
            cluster_indices[j] = representative_indices.index(i)
    for file in [tmp_output + '_rep_seq.fasta', tmp_output + '_all_seqs.fasta', tmp_output + '_cluster.tsv']:
        os.remove(file)
    return np.array(cluster_indices), np.array(representative_indices)


def calculate_weights(list_sequences, resolutions=[100, 95, 90, 70], coverage=0.8, covmode='0'):
    list_sequences = np.array(list_sequences, dtype=str)
    N = len(list_sequences)
    nresolutions = len(resolutions)
    hierarchical_cluster_indices = [np.arange(N)]
    hierarchical_representative_indices = [np.arange(N)]
    hierarchical_representative_sequences = [list_sequences]
    hierarchical_cluster_sizes = [np.ones(N)]

    for k, resolution in enumerate(resolutions):
        cluster_indices, representative_indices = cluster_sequences(hierarchical_representative_sequences[k],
                                                                    seqid=resolution / 100,
                                                                    coverage=coverage, covmode=covmode)
        cluster_sizes = np.array([(cluster_indices == k).sum() for k in range(len(representative_indices))])
        representative_sequences = hierarchical_representative_sequences[k][representative_indices]
        hierarchical_cluster_indices.append(cluster_indices)
        hierarchical_representative_indices.append(representative_indices)
        hierarchical_representative_sequences.append(representative_sequences)
        hierarchical_cluster_sizes.append(cluster_sizes)

    hierarchical_num_clusters = [len(representative_sequences) for representative_sequences in
                                 hierarchical_representative_sequences]
    hierarchical_weights = [np.ones(hierarchical_num_clusters[-1])]
    for k in range(1, nresolutions + 1)[::-1]:
        num_neighbours = 1.0 / hierarchical_cluster_sizes[k]
        weights = (hierarchical_weights[-1] * num_neighbours)[hierarchical_cluster_indices[k]]
        hierarchical_weights.append(weights)
    hierarchical_weights = hierarchical_weights[::-1]
    return hierarchical_weights[0]

def change_model_num_for_header(input_file,output_file,header,model_num,table_path):

    with open(input_file, 'r') as infile:
        in_header = False
        new_lines = []
        for line in infile:
            if line.startswith('>'):
                if line[:-1] == header:
                    in_header = True
                    models_chains = line.split('_')[1].split('+')
                    updated_models_chains = [f'{str(model_num)}{model_chain[1:]}' for model_chain in models_chains]
                    line = f'{line.split("_")[0]}_{"+".join(updated_models_chains)}'
                else:
                    in_header = False          
            else:
                if in_header:
                    line = f"{str(model_num)} {line[2:]}"
            new_lines.append(line)
        with open(output_file, 'w') as outfile:
            outfile.writelines(new_lines)

def update_pdb_id_and_get_fold(table_path, pdb_id, new_model_num):
    # Read the table into a DataFrame
    df = pd.read_csv(table_path)
    
    # Find the row with the given PDB ID
    row = df[df['PDB ID'] == pdb_id]
    
    if row.empty:
        raise ValueError(f"PDB ID {pdb_id} not found in the table.")
    
    # Get the fold number
    fold_number = int(row['Set'].values[0][-1])
    
    # Update the PDB ID with the new model number
    models_chains = pdb_id.split('_')[1].split('+')
    updated_models_chains = [f'{new_model_num}{model_chain[1:]}' for model_chain in models_chains]
    new_pdb_id = f'{pdb_id.split("_")[0]}_{"+".join(updated_models_chains)}'
    
    # Update the DataFrame
    df.loc[df['PDB ID'] == pdb_id, 'PDB ID'] = new_pdb_id
    
    # Save the updated table back to the file
    df.to_csv(table_path, index=False)
    
    return fold_number

def delete_short_chains(input_file,out_file,min_number_of_residues):
    in_chain = False
    with open(out_file, 'w') as out_file:
        with open(input_file, 'r') as infile:
            chain_lines = []
            for line in infile:
                if line.startswith('>'):
                    if in_chain and len(chain_lines) > (min_number_of_residues + 1):
                        out_file.writelines(chain_lines)
                    elif in_chain:
                        print(f' chain {chain_lines[0]} has {len(chain_lines)-1} residues and deleted')
                    in_chain = True
                    chain_lines = [line]
 
                else:
                    chain_lines.append(line)
            if in_chain and len(chain_lines) > (min_number_of_residues + 1):
                out_file.writelines(chain_lines)
    # ! that can be a unique special case where couple of chains are under the min_number_of_residues but are toghether under the same header"

def create_table(dataset_folder,output_folder):
    all_origins = []
    all_folds = []
    all_weights = []
    all_sequences = []
    for k in range(1, 6):
        dataset_file = os.path.join(dataset_folder, f'labels_fold{k}.txt' )
        list_origins, list_sequences, list_resids, list_labels = read_labels(dataset_file)
        all_origins += list(list_origins)
        all_folds += ['Fold %s' % k] * len(list_origins)
        all_sequences += list(list_sequences)

    all_origins = np.array(all_origins)
    all_folds = np.array(all_folds)
    all_sequences = np.array(all_sequences)

    all_weights_v0 = np.ones(len(all_sequences))
    all_weights_v1 = calculate_weights(all_sequences, resolutions=[100, 95, 90, 70])
    all_weights_v2 = calculate_weights(all_sequences, resolutions=[95])
    all_weights_v3 = calculate_weights(all_sequences, resolutions=[70])

    table = pd.DataFrame({
        'PDB ID': all_origins,
        'Length': [len(sequence) for sequence in all_sequences],
        'Set': all_folds,
        'Sample weight': all_weights_v1,
        'Sample weight none': all_weights_v0,
        'Sample weight flat95': all_weights_v2,
        'Sample weight flat70': all_weights_v3,
    })
    table.to_csv(os.path.join(output_folder, f'table.csv'))
    print(f' the table was created in {os.path.join(output_folder, f"table.csv")}')

# %%

def manual_fixes(output_folder):
    # 1q0w_0-A -> 1q0w_1-A
    table_path = os.path.join(output_folder, 'table.csv')
    fold_num = update_pdb_id_and_get_fold(table_path, '1q0w_0-A', 1)
    input_file = os.path.join(output_folder, f'labels_fold{fold_num}.txt')
    change_model_num_for_header(input_file,input_file,'>1q0w_0-A',1,table_path) 

def remove_keys(file_path):
    # ONLY for in order to fix the replicating dataset
    with open(file_path, 'r') as file:
        content = file.read()

    blocks = content.split('>')
    new_content = []

    for block in blocks:
        lines = block.splitlines()
        if lines and ('_1' in lines[0] or '+1' in lines[0]):
            continue
        if block.strip():  # Ensure we don't add empty blocks
            new_content.append('>' + block)

    with open(file_path, 'w') as file:
        file.write(''.join(new_content))

def create_sample_datasets_and_table(with_scanNet_addition:bool,with_augmentations_addition:bool,datasets_dir:str):
    for i in range(5):
        fold_file = os.path.join(datasets_dir, f'labels_fold{i+1}{with_scanNet_addition}{with_augmentations_addition}.txt')
        sample_file = fold_file.split('.txt')[0] + '_sample.txt'
        sample_examples = {}  # keys: 1 (contains '|'), 2 (contains '+' only), 3 (others)

        # Read all lines and split into examples
        with open(fold_file, 'r') as f:
            all_examples = f.read().split(">")

        for example in all_examples:
            header = example.split('\n')[0]
            if header == '':
                continue
            if '|' in header:
                type_key = 1
            elif '+' in header:
                type_key = 2
            else:
                type_key = 3
            if type_key not in sample_examples:
                sample_examples[type_key] = example

        fold_examples = list(sample_examples.values())
        sample_fold_info = ">"+">".join(fold_examples)

        with open(sample_file,'w') as f:
            f.write(sample_fold_info)

    create_table(datasets_dir, datasets_dir, with_augmentations_addition, with_scanNet_addition, '_sample')

if __name__ == '__main__':
    plan_dict = {
        'name': "v2",
        'seq_id': "0.95",
        'ASA_THRESHOLD_VALUE': 0.1,
        'weight_bound_for_ubiq_fold':0.21,
        'add_model_num_to_dataset': True,
        'create_table': True,
        }
    
    dir_name = os.path.join(paths.scanNet_data_for_training_path,plan_dict['name'])
    os.makedirs(dir_name, exist_ok=True)

    name = f'seq_id_{plan_dict["seq_id"]}_asaThreshold_{plan_dict["ASA_THRESHOLD_VALUE"]}_bound_{plan_dict["weight_bound_for_ubiq_fold"]}'
    output_folder = os.path.join(dir_name,name)


    PSSM_path = os.path.join(paths.PSSM_path, plan_dict['name'])
    PSSM_seq_id_folder = os.path.join(PSSM_path,f'seq_id_{plan_dict["seq_id"]}_asaThreshold_{plan_dict["ASA_THRESHOLD_VALUE"]}')
    bound_addition = f"_{str(plan_dict['weight_bound_for_ubiq_fold']).split('.')[1]}"

    if plan_dict['add_model_num_to_dataset']:
        for i in range(5):
            input_file = os.path.join(PSSM_seq_id_folder,f'PSSM{bound_addition}_{i}_with_augmentations.txt' )
            output_deletion_file = os.path.join(output_folder,f'PSSM{bound_addition}_{i}_with_augmentations_deleted_short_chains.txt')
            delete_short_chains(input_file, output_deletion_file, 15)
            output_file = os.path.join(output_folder, f'labels_fold{i+1}.txt')
            add_model_num_to_dataset(output_deletion_file, output_file)

    #ONLY FOR REPLICATION 
    # for i in range(5):
    #     remove_keys(os.path.join(output_folder, f'labels_fold{i+1}.txt'))
    
    if plan_dict['create_table']:
        create_table(output_folder,output_folder)
    # create_sample_datasets_and_table(with_augmentations_addition = with_augmentations_addition,
    #     with_scanNet_addition=with_scanNet_addition,datasets_dir = output_folder)
    
    # manual_fixes(output_folder)

    #try delete short chains
    # %%

    # dataset_file = 'FullPssmContent'
    # all_origins, all_sequences, all_resids, all_labels = read_labels(dataset_file)

    # input_file =  os.path.join(os.path.join(output_folder, f'labels_fold{3}.txt'))
    # output_file =  os.path.join(os.path.join(output_folder, f'labels_fold{3}.txt'))
    # change_model_num_for_header(input_file,output_file,'>6o82_1-A',0) 

# 