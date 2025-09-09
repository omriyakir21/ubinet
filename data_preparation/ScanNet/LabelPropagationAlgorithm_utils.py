import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import subprocess
import paths
import numpy as np
from models.ScanNet_Ub.preprocessing.sequence_utils import load_FASTA, num2seq
from data_preparation.ScanNet.db_creation_scanNet_utils import save_as_pickle,aa_out_of_chain,load_as_pickle,get_str_seq_of_chain
import random
from Bio.PDB.PDBParser import PDBParser
import gzip
from data_preparation.ScanNet.aggregate_scanNet_multiclass import parse_labels
def create_chain_dict_with_all_info_only_pssm(chains_sequences, chains_labels, chain_names, lines):
    chain_dict = {}
    for i in range(len(chain_names)):
        id  =chain_names[i]
        id = id.split('_')[0] + '_' + id.split('_')[1].split('$')[-1]
        chain_dict[id] = {}
        chain_dict[id]['sequences'] = chains_sequences[i]
        chain_dict[id]['labels'] = chains_labels[i]
        chain_dict[id]['lines'] = lines[i]
    return chain_dict


def split_receptors_into_individual_chains(pssm_content_file_path, asa_pssm_content_file_path):
    f = open(pssm_content_file_path, 'r')
    fAsa = open(asa_pssm_content_file_path, 'r')
    lines = f.readlines()
    asa_lines = fAsa.readlines()
    chains_keys = []
    chains_sequences = []
    chains_labels = []
    chains_asa_values = []
    chain_names = []
    chain_key = lines[0][1:-1]
    chains_keys.append(chain_key)
    chain_seq = ''
    chain_labels = []
    chain_asa_values = []
    chain_name = None
    for i in range(1, len(lines)):
        try:
            line = lines[i]
            asa_line = asa_lines[i]
            if line[0] == '>':
                chains_sequences.append(chain_seq)
                chains_asa_values.append(chain_asa_values)
                chains_labels.append(chain_labels)
                assert len(chain_asa_values) == len(chain_labels)
                chain_seq = ''
                chain_asa_values = []
                chain_labels = []
                chain_key = line[1:-1]
                chains_keys.append(chain_key)
                continue
            elif chains_keys[len(chains_keys) - 1] + '$' + line.split(" ")[0] != chain_name:
                if len(chain_seq) > 0:
                    chains_sequences.append(chain_seq)
                    chains_labels.append(chain_labels)
                    chains_asa_values.append(chain_asa_values)
                    chain_seq = ''
                    chain_asa_values = []
                    chain_labels = []
                chain_name = chains_keys[len(chains_keys) - 1] + '$' + line.split(" ")[0]
                chain_names.append(chain_name)
            
            asa_info = asa_line.split(" ")
            amino_acid_info = line.split(" ")
            chain_seq += (amino_acid_info[2])
            chain_labels.append(amino_acid_info[3][:-1])
            chain_asa_values.append(float(asa_info[3][:-1]))
        except Exception as e:
            print(f"Error processing line {i}: {line.strip()}")
            print(f"ASA line: {asa_line.strip()}")
            raise e

    chains_sequences.append(chain_seq)
    chains_labels.append(chain_labels)
    chains_asa_values.append(chain_asa_values)
    assert (len(chain_names) == len(chains_sequences) == len(chains_labels) == len(chains_asa_values))
    f.close()
    return chains_keys, chains_sequences, chains_labels, chain_names, lines, chains_asa_values

def split_receptors_into_individual_chains_only_PSSM(pssm_content_file_path):
    f = open(pssm_content_file_path, 'r')
    lines = f.readlines()
    chains_keys = []
    chains_sequences = []
    chains_labels = []
    chain_names = []
    chain_key = lines[0][1:-1]
    chains_keys.append(chain_key)
    chain_seq = ''
    chain_labels = []
    chain_name = None
    for i in range(1, len(lines)):
        line = lines[i]
        if line[0] == '>':
            chains_sequences.append(chain_seq)
            chains_labels.append(chain_labels)
            chain_seq = ''
            chain_labels = []
            chain_key = line[1:-1]
            chains_keys.append(chain_key)
            continue
        elif chains_keys[len(chains_keys) - 1] + '$' + line.split(" ")[0] != chain_name:
            if len(chain_seq) > 0:
                chains_sequences.append(chain_seq)
                chains_labels.append(chain_labels)
                chain_seq = ''
                chain_labels = []
            chain_name = chains_keys[len(chains_keys) - 1] + '$' + line.split(" ")[0]
            chain_names.append(chain_name)

        if line[-1]!="\n":
            line+="\n"
        amino_acid_info = line.split(" ")
        chain_seq += (amino_acid_info[2])
        chain_labels.append(amino_acid_info[3][:-1])

    chains_sequences.append(chain_seq)
    chains_labels.append(chain_labels)
    assert (len(chain_names) == len(chains_sequences) == len(chains_labels))
    f.close()
    return chains_keys, chains_sequences, chains_labels, chain_names, lines

def create_cluster_participants_indexes(cluster_indexes):
    clusters_participants_list = []
    for i in range(np.max(cluster_indexes) + 1):
        clusters_participants_list.append(np.where(cluster_indexes == i)[0])
    return clusters_participants_list


def aggregate_cluster_sequences(chains_sequences, clusters_participants_list, index):
    sequences = [chains_sequences[i] for i in clusters_participants_list[index]]
    return sequences


def apply_mafft(sequences, mafft, go_penalty=1.53,
                ge_penalty=0.0, name=None, numeric=False, return_index=True, high_accuracy=True):
    if name is None:
        name = '%.10f' % np.random.rand()
    input_file = os.path.join(paths.tmp_path, 'tmp_%s_unaligned.fasta' % name)
    output_file = os.path.join(paths.tmp_path, 'tmp_%s_aligned.fasta' % name)
    instruction_file = os.path.join(paths.tmp_path, 'tmp_%s.sh' % name)
    with open(input_file, 'w') as f:
        for k, sequence in enumerate(sequences):
            f.write('>%s\n' % k)
            f.write(sequence + '\n')
    if high_accuracy:
        command = '%s  --amino --localpair --maxiterate 1000 --op %s --ep %s %s > %s' % (
            mafft, go_penalty, ge_penalty, input_file, output_file)
    else:
        command = '%s  --amino --auto --op %s --ep %s %s > %s' % (
            mafft, go_penalty, ge_penalty, input_file, output_file)
    print(command)
    with open(instruction_file, 'w') as f:
        f.write(command)
    os.system('sh %s' % instruction_file)

    alignment = load_FASTA(
        output_file, drop_duplicates=False)[0]
    if return_index:
        is_gap = alignment == 20
        index = np.cumsum(1 - is_gap, axis=1) - 1
        index[is_gap] = -1

    if not numeric:
        alignment = num2seq(alignment)
    os.system('rm %s' % input_file)
    os.system('rm %s' % output_file)
    os.system('rm %s' % instruction_file)

    if return_index:
        return alignment, index
    else:
        return alignment


def apply_mafft_for_all_clusters(chains_sequences, clusters_participants_list, path_to_mafft_exec):
    clusters_dict = dict()
    aligments = []
    indexes = []
    try:
        for i in range(len(clusters_participants_list)):
            sequences = aggregate_cluster_sequences(chains_sequences, clusters_participants_list, i)
            aligment, index = apply_mafft(sequences, path_to_mafft_exec)
            aligments.append(aligment)
            indexes.append(index)
    except Exception as e:
        print(f"Error occurred while applying MAFFT: {e}\n i = {i}, sequences = {sequences}")
        raise(e)
    clusters_dict['aligments'] = aligments
    clusters_dict['indexes'] = indexes
    return clusters_dict

def calculate_consensus_labels(msa_length:int,number_of_participants:int,label_length,labels_after_aligment:list):
    consensus_list = []
    for j in range(msa_length):
        consensus = ['10' for _ in range(label_length // 2)] 
        for i in range(number_of_participants):
            for k in range(0, label_length, 2):
                if labels_after_aligment[i][j][k:k + 2] == '01':
                    consensus[k // 2] = '01'
        consensus = "".join(consensus)
        consensus_list.append(consensus)
    return consensus_list

def create_propagated_labels_for_cluster(index, chains_labels, cluster_participants_list, chains_asa_values,
                                         asa_threshold_value):
    number_of_participants = index.shape[0]
    msa_length = index.shape[1]
    label_length = len(chains_labels[0][0])
    assert (number_of_participants == len(cluster_participants_list))
    new_labels = [[] for _ in range(number_of_participants)]
    labels_after_aligment = [['10'*(label_length//2) for _ in range(msa_length)] for _ in range(number_of_participants)]
    for i in range(number_of_participants):
        current_labels = chains_labels[cluster_participants_list[i]]
        indexs_of_parcipitant = index[i]
        for j in range(msa_length):
            if indexs_of_parcipitant[j] != -1:  # not a gap
                labels_after_aligment[i][j] = current_labels[indexs_of_parcipitant[j]]

    consensus_list = calculate_consensus_labels(msa_length=msa_length,
                                            number_of_participants=number_of_participants,
                                            label_length=label_length,
                                            labels_after_aligment=labels_after_aligment)

    column_max_asa_value = []
    for j in range(msa_length):
        max_asa = 0
        for i in range(number_of_participants):
            chain_index = cluster_participants_list[i]
            participant_chain_asa_values =chains_asa_values[chain_index] 
            print(len(participant_chain_asa_values))
            index_of_participant = index[i]
            if index_of_participant[j] != -1:
                max_asa = max(max_asa,participant_chain_asa_values[index_of_participant[j]])
        column_max_asa_value.append(max_asa)
    column_thresholds = [min(asa_threshold_value, 0.75 * column_max_asa_value[j]) for j in range(msa_length)]
    for i in range(number_of_participants):
        chain_index = cluster_participants_list[i]
        indexs_of_parcipitant = index[i]
        # threshold = min(asa_threshold_value, 0.75 * max(chains_asa_values[chain_index]))
        for j in range(msa_length):
            if indexs_of_parcipitant[j] != -1:  # not a gap
                if chains_asa_values[chain_index][len(new_labels[i])] > column_thresholds[j]:
                    new_labels[i].append(consensus_list[j])
                else:
                    new_labels[i].append(chains_labels[chain_index][len(new_labels[i])])
    return new_labels


def find_chain_names_for_cluster(clusters_participants_list, chain_names, i):
    cluster_chains_names = [chain_names[j] for j in clusters_participants_list[i]]
    return cluster_chains_names


def find_chain_names_for_clusters(clusters_participants_list, chain_names):
    print(chain_names)
    clusters_chains_names = [find_chain_names_for_cluster(clusters_participants_list, chain_names, i) for i in
                             range(len(clusters_participants_list))]
    return clusters_chains_names


def create_propagated_pssm_file(clusters_dict, chains_labels, clusters_participants_list,
                                chains_sequences, chain_names, lines, chains_asa_values, propagated_file_path,
                                asa_threshold_value):
    num_of_clusters = len(clusters_dict['indexes'])
    num_of_chains = len(chains_sequences)
    new_labels = [None for i in range(num_of_chains)]
    clusters_chains_names = find_chain_names_for_clusters(clusters_participants_list, chain_names)
    print(f'clusters_chains_names: {clusters_chains_names}')
    # clustersIndexes = [findIndexesForCluster(clusterChainNames) for clusterChainNames in clusters_chains_names]
    for i in range(num_of_clusters):
        cluster_new_labels = create_propagated_labels_for_cluster(clusters_dict['indexes'][i], chains_labels,
                                                                  clusters_participants_list[i], chains_asa_values,
                                                                  asa_threshold_value)
        for j in range(len(clusters_participants_list[i])):
            new_labels[clusters_participants_list[i][j]] = cluster_new_labels[j]

    propagated_file = open(propagated_file_path, 'w')
    chain_index = -1
    chain_name = None
    for line in lines:
        if line[0] == '>':
            chains_key = line[1:-1]
        else:
            if chains_key + '$' + line.split(" ")[0] != chain_name:
                chain_name = chains_key + '$' + line.split(" ")[0]
                chain_index += 1
                amino_acid_num = 0
            splited_line = line.split(" ")
            splited_line[-1] = str(new_labels[chain_index][amino_acid_num]) + '\n'
            line = " ".join(splited_line)
            amino_acid_num += 1
        propagated_file.write(line)
    propagated_file.close()


def create_quantile_asa_dicts(lines):
    amino_acid_asa_dict = dict()
    for line in lines:
        if line[0] != '>':
            splitted_line = line.split(" ")
            asa_val = splitted_line[3][:-1]
            amino_acid_char = splitted_line[2]
            if amino_acid_char not in amino_acid_asa_dict:
                amino_acid_asa_dict[amino_acid_char] = []
            amino_acid_asa_dict[amino_acid_char].append(float(asa_val))
    quentile_asa_amino_acid_dict = dict()

    for amino_acid_char in amino_acid_asa_dict.keys():
        quantile5 = np.percentile(amino_acid_asa_dict[amino_acid_char], 5)
        quantile95 = np.percentile(amino_acid_asa_dict[amino_acid_char], 95)
        quentile_asa_amino_acid_dict[amino_acid_char] = (quantile5, quantile95)
    return quentile_asa_amino_acid_dict


def normalize_value(current_val, quantile5, quantile95):
    if current_val <= quantile5:
        return 0
    if current_val >= quantile95:
        return 1
    normalize_value = (current_val - quantile5) / (quantile95 - quantile5)
    return normalize_value


def normalize_asa_data(full_asa_pssm_path, normalized_asa_path,quantile_amino_acids_dict_path):
    f = open(full_asa_pssm_path, 'r')
    lines = f.readlines()
    f.close()
    if not os.path.exists(quantile_amino_acids_dict_path):
        quentile_asa_amino_acid_dict = create_quantile_asa_dicts(lines)
        save_as_pickle(quentile_asa_amino_acid_dict, quantile_amino_acids_dict_path)
    else:
        quentile_asa_amino_acid_dict = load_as_pickle(quantile_amino_acids_dict_path)

    normalize_asa_pssm_content_file = open(normalized_asa_path, 'w')
    for line in lines:
        if line[0] == '>':
            normalize_asa_pssm_content_file.write(line)
        else:
            splitted_line = line.split(" ")
            asa_val = float(splitted_line[3][:-1])
            amino_acid_char = splitted_line[2]
            normalized_asa_value = normalize_value(asa_val, quentile_asa_amino_acid_dict[amino_acid_char][0],
                                                   quentile_asa_amino_acid_dict[amino_acid_char][1])
            splitted_line[3] = str(normalized_asa_value) + '\n'
            new_line = " ".join(splitted_line)
            normalize_asa_pssm_content_file.write(new_line)
    normalize_asa_pssm_content_file.close()

def parse_gzipped_pdb(file_path):
    # Initialize the PDB parser
    parser = PDBParser(QUIET=True)
    
    # Open the gzipped file
    with gzip.open(file_path, 'rt') as gz_file:
        # Parse the structure
        structure = parser.get_structure('structure_id', gz_file)
    
    return structure

def create_asa_file(input_parts_folder:str,scanNet_pssm_parts_folder:str,
                    scanNet_asa_parts_folder:str,scanNet_not_included_parts_folder:str,
                    index:str,pdb_dict:dict):
    pssm_file = os.path.join(input_parts_folder, f'part_{index}.txt')
    asa_output_path = os.path.join(scanNet_asa_parts_folder, f'part_{index}_asa.txt')
    pssm_output_path = os.path.join(scanNet_pssm_parts_folder, f'part_{index}_pssm.txt')
    pdbs_not_included_path = os.path.join(scanNet_not_included_parts_folder, f'part_{index}_not_included.txt')
    from Bio.PDB.MMCIFParser import MMCIFParser
    from Bio.PDB.SASA import ShrakeRupley
    structures_folder = '/home/iscb/wolfson/jeromet/Data/PDB_files/'
    cif_parser = MMCIFParser(QUIET=True)
    with open(pssm_file, 'r') as f:
        lines = f.readlines()
    asa_lines = []
    pssm_lines = []
    not_included_lines = []
    add_asa_lines = []
    add_pssm_lines = []
    problematic_pdb = False
    for line in lines:
        if line[0] == '>':#>1a2n_0-A
            asa_lines.extend(add_asa_lines)
            pssm_lines.extend(add_pssm_lines)
            add_asa_lines = []
            add_pssm_lines = []
            problematic_pdb = False
            try:
                print(f'Processing line: {line.strip()}')
                header = line.strip()
                pdb = line[1:5]
                model = int(line[6])
                chain_string = line[8:-1]
                if header in pdb_dict:
                    structure_path = pdb_dict[header]
                    structure = parse_gzipped_pdb(structure_path)
                else:          
                    structure_path =  f'{os.path.join(structures_folder,pdb+".cif")}'
                    if os.path.exists(structure_path):
                        structure = cif_parser.get_structure(pdb, structure_path)
                    else:
                        print(f"File {structure_path} not found, downloading...")
                        subprocess.run(['wget', '-q', '-O', structure_path, f'https://files.rcsb.org/download/{pdb}.cif'])
                        if os.path.exists(structure_path):
                            structure = cif_parser.get_structure(pdb, structure_path)
                        else:
                            raise FileNotFoundError(f"Could not find or download structure for {pdb}.")
                # parse and get the model
                model = structure[model]
                # get the chain
                sasa_calc = ShrakeRupley()
                for chain in model:
                    print(f'Processing Chain: {chain.id}')
                    if chain.id == chain_string:
                        break
                if chain.id != chain_string:
                    raise ValueError(f"Chain {chain_string} not found in PDB {pdb}, Model {model}.")
                add_asa_lines.append(line)
                add_pssm_lines.append(line)
            except Exception as e:
                print(f"Error processing PDB {pdb}, Model {model}, Chain {chain_string}: {e}")
                problematic_pdb = True
                not_included_lines.append(line)
                add_asa_lines = []
                add_pssm_lines = []
                continue
        elif not problematic_pdb:
            try:
                splitted_line = line.split(" ")
                aa_index = int(splitted_line[1])
                aa = chain[(' ', aa_index, ' ')]  # get the amino acid with this index
                sasa_calc.compute(aa, level='R')
                asa_val = aa.sasa
                splitted_line[3] = str(asa_val) + '\n'
                add_asa_lines.append(" ".join(splitted_line))
                add_pssm_lines.append(line)
            except Exception as e: 
                problematic_pdb = True
                not_included_lines.append(add_pssm_lines[0])
                add_asa_lines = []
                add_pssm_lines = []
                continue
        
    asa_lines.extend(add_asa_lines)
    pssm_lines.extend(add_pssm_lines)
    with open(asa_output_path, 'w') as asa_file:
        asa_file.writelines(asa_lines)
    with open(pssm_output_path, 'w') as pssm_file:
        pssm_file.writelines(pssm_lines)
    if len(not_included_lines) > 0:
        with open(pdbs_not_included_path, 'w') as not_included_file:
            not_included_file.writelines(not_included_lines)
    else:
        # write empty file
        with open(pdbs_not_included_path, 'w') as not_included_file:
            not_included_file.write('')

def concat_files_with_newline_seperator(pssm_files, output_file):
    with open(output_file, 'w') as outfile:
        for pssm_file in pssm_files:
            with open(pssm_file, 'r') as infile:
                content = infile.read()
                #add a new line if not already present
                if content and not content.endswith('\n'):
                    content += '\n'
                outfile.write(content)

def propagate_labels_of_chain(ubiq_chain_labels,scanNet_chain_labels):
    assert len(ubiq_chain_labels) == len(scanNet_chain_labels)
    new_labels = []
    for i in range(len(ubiq_chain_labels)):
        new_label = ''
        for j in range(0,len(ubiq_chain_labels[i]),2):
            if ubiq_chain_labels[i][j:j+2]== '01' or scanNet_chain_labels[i][j:j+2] == '01':
                new_label += '01'
            else:
                new_label += '10'
        new_labels.append(new_label)
    # new_labels = [str(max(int(ubiq_chain_labels[i]),int(scanNet_chain_labels[i]))) for i in range (len(ubiq_chain_labels))]
    return new_labels

def recreate_PSSM_file_from_dict(chain_dict,PSSM_file, output_file_path):
    with open(PSSM_file, 'r') as f:
        lines = f.readlines()
    with open(output_file_path, 'w') as f:
        for line in lines:
            try:
                if line[0] == '>':
                    full_name = line[1:len(line) - 1]
                    pdb = full_name[0:4]
                    amino_acid_cnt = 0
                    save_chain = ''
                    model_num = full_name[5]
                    chains = full_name.split('_')[1]
                    chains = chains.split('+')
                    chains = [c[2:] for c in chains]  # remove the 'A_' prefix
                    valid_chains = [c for c in chains if f'{pdb}_{c}' in chain_dict.keys()]
                    line_to_write = f'>{pdb}_{"+".join(model_num + "-" + c for c in valid_chains)}\n'
                    if not valid_chains:
                        continue
                    f.write(line_to_write)
                else:
                    line = line.split(' ')
                    chain = line[0]
                    if chain not in valid_chains:
                        continue
                    if chain != save_chain:
                        save_chain = chain
                        amino_acid_cnt = 0

                    line[3] = chain_dict[f'{pdb}_{chain}']['labels'][amino_acid_cnt]
                    amino_acid_cnt += 1
                    line = " ".join(line)+'\n'
                    f.write(line)
            except Exception as e:
                raise Exception(f"Error processing line: {line}. Error: {e}")

def keep_only_chars(string):
    return ''.join([char for char in string if char.isalpha()])

def list_creation_util(full_name):
    pdb_name = full_name[0:4]
    chains_string = full_name.split('_')[1]
    chains_strings_list = chains_string.split('+')
    chains_names_list = [keep_only_chars(chainString) for chainString in chains_strings_list]
    pdb_names_with_chains_list = [pdb_name + chainName for chainName in chains_names_list]
    return pdb_name, pdb_names_with_chains_list

def list_creation(file_name):
    """
    :param file_name: PSSM file
    :return: tuple(names_list,sizes_list)
    names_list = list of all the chains's name in the file
    sizes_list = list of all the chains's number of amino acids in the file
    """
    names_list = []
    full_names_list = []
    pdb_names_with_chains_lists = []
    sizes_list = []
    sequence_lists = [[]]
    file1 = open(file_name, 'r')
    last_chain_name = ''
    line = file1.readline().split()
    cnt = 0
    seq = ''
    while len(line) != 0:  # end of file
        cnt += 1
        if len(line) == 1:  # in chain header line
            sequence_lists[len(sequence_lists) - 1].append(seq)
            sizes_list.append(cnt)
            full_name = line[0][1:]
            full_names_list.append(full_name)
            pdb_name, pdb_names_with_chains_list = list_creation_util(full_name)
            names_list.append(pdb_name)
            pdb_names_with_chains_lists.append(pdb_names_with_chains_list)
            try:
                if len(pdb_names_with_chains_lists) > 1:
                    assert (len(sequence_lists[len(sequence_lists) - 1]) == len(
                        pdb_names_with_chains_lists[len(pdb_names_with_chains_lists) - 2]))
                    assert (sizes_list[len(sizes_list) - 1]) == sum(
                        [len(seq) for seq in sequence_lists[len(sequence_lists) - 1]])
            except:
                raise Exception(f"pdb name: {pdb_name},pdb_names_with_chains_list : {pdb_names_with_chains_list}, sequence_lists: {sequence_lists[len(sequence_lists) - 1]}, sizes_list: {sizes_list[len(sizes_list) - 1]}")
            sequence_lists.append([])
            cnt = -1
            seq = ''
            last_chain_name = ''
        else:
            if last_chain_name != line[0]:  # switching chains
                last_chain_name = line[0]
                if len(seq) != 0:
                    sequence_lists[len(sequence_lists) - 1].append(seq)
                seq = ''
            seq = seq + line[2]  # not chain's name
        line = file1.readline().split()
    sizes_list.append(cnt)
    sequence_lists[len(sequence_lists) - 1].append(seq)
    sizes_list = sizes_list[1:]  # first one is redundent
    sequence_lists = sequence_lists[1:]
    file1.close()
    return names_list, sizes_list, sequence_lists, full_names_list, pdb_names_with_chains_lists

def propagate_labels_and_concat(ubiq_PSSM:str,scanNet_PSSM:str,normalized_asa_scanNet_file:str, ubiq_output_PSSM:str, scanNet_output_PSSM:str,scanNet_output_asa:str):
    print(f"Processing Ubiq PSSM file: {ubiq_PSSM}")
    print(f"Processing ScanNet PSSM file: {scanNet_PSSM}")
    print(f"Processing normalized ASA ScanNet file: {normalized_asa_scanNet_file}")
    
    _, _, _, scanNet_full_names_list,_ = list_creation(scanNet_PSSM)
    if len(scanNet_full_names_list) != len(set(scanNet_full_names_list)):
        raise ValueError("Duplicate names found in full_names_list.")
    
    print(f"Number of unique full names in ScanNet: {len(set(scanNet_full_names_list))}")
    _, ubiq_chains_sequences, ubiq_chains_labels, ubiq_chain_names, ubiq_lines = split_receptors_into_individual_chains_only_PSSM(ubiq_PSSM)
    _, scanNet_chains_sequences, scanNet_chains_labels, scanNet_chain_names, scanNet_lines = split_receptors_into_individual_chains_only_PSSM(scanNet_PSSM)
    _,scanNet_asa_chains_sequecnes, scanNet_asa_chains_labels, scanNet_asa_chain_names, scanNet_asa_lines = split_receptors_into_individual_chains_only_PSSM(normalized_asa_scanNet_file)
    ubiq_chain_dict = create_chain_dict_with_all_info_only_pssm(ubiq_chains_sequences, ubiq_chains_labels, ubiq_chain_names, ubiq_lines)
    scanNet_chain_dict = create_chain_dict_with_all_info_only_pssm(scanNet_chains_sequences, scanNet_chains_labels, scanNet_chain_names, scanNet_lines)
    scanNet_chain_dict_asa = create_chain_dict_with_all_info_only_pssm(scanNet_asa_chains_sequecnes, scanNet_asa_chains_labels, scanNet_asa_chain_names, scanNet_asa_lines)
    common_keys = set(ubiq_chain_dict.keys()).intersection(set(scanNet_chain_dict.keys()))
    for key in common_keys:
        ubiq_chain_labels = ubiq_chain_dict[key]['labels']
        scanNet_chain_labels = scanNet_chain_dict[key]['labels']
        try:
            new_labels = propagate_labels_of_chain(ubiq_chain_labels, scanNet_chain_labels)
        except Exception as e:
            raise ValueError(f"Error propagating labels for key {key}: {e}")
        
        ubiq_chain_dict[key]['labels'] = new_labels
        # remove the scanNet key from scanNet_chain_dict
        del scanNet_chain_dict[key]
        del scanNet_chain_dict_asa[key]
    recreate_PSSM_file_from_dict(scanNet_chain_dict,scanNet_PSSM,scanNet_output_PSSM)
    print(f"Recreated PSSM file from dict for ScanNet: {scanNet_output_PSSM}")
    recreate_PSSM_file_from_dict(ubiq_chain_dict,ubiq_PSSM,ubiq_output_PSSM)
    print(f"Recreated PSSM file from dict for Ubiq: {ubiq_output_PSSM}")
    recreate_PSSM_file_from_dict(scanNet_chain_dict_asa,normalized_asa_scanNet_file,scanNet_output_asa)
    print(f"Recreated ASA file from dict for ScanNet: {scanNet_output_asa}")
    return ubiq_chain_dict, scanNet_chain_dict, scanNet_chain_dict_asa

def propagate_for_duplicate_sequences(PSSM_file:str,asa_file:str,output_PSSM_file:str,output_asa_file:str):
    _, chains_sequences, chains_labels, chain_names, lines = split_receptors_into_individual_chains_only_PSSM(PSSM_file)
    _, chains_asa_sequences, chains_asa_labels, chain_names_asa, asa_lines = split_receptors_into_individual_chains_only_PSSM(asa_file)
    chain_dict = create_chain_dict_with_all_info_only_pssm(chains_sequences, chains_labels, chain_names, lines)
    chain_dict_asa = create_chain_dict_with_all_info_only_pssm(chains_asa_sequences, chains_asa_labels, chain_names_asa, asa_lines)
    sequence_to_chain_dict = {}
    print(f"chain names 10 examlpes {list(chain_dict.keys())[:10]}")
    print(f"len chain names :{len(list(chain_dict.keys()))}")
    for chain_name, chain_info in chain_dict.items():
        sequence = chain_info['sequences']
        if sequence not in sequence_to_chain_dict:
            sequence_to_chain_dict[sequence] = []
        sequence_to_chain_dict[sequence].append(chain_name)
    
    for chain_names in sequence_to_chain_dict.values():
        if len(chain_names) > 1:
            # If there are multiple chains with the same sequence, propagate labels
            chain_name = chain_names[0]
            new_labels = chain_dict[chain_name]['labels']
            for new_chain_name in chain_names[1:]:
                add_labels = chain_dict[new_chain_name]['labels']
                new_labels = propagate_labels_of_chain(new_labels, add_labels)
            for chain_name in chain_names:
                chain_dict[chain_name]['labels'] = new_labels

    recreate_PSSM_file_from_dict(chain_dict, PSSM_file, output_PSSM_file)
    recreate_PSSM_file_from_dict(chain_dict_asa, asa_file, output_asa_file)

def divide_pssm_to_examples(pssm_file:str):
    with open(pssm_file, 'r') as f:
        all_file = f.read() 
        examples = all_file.split('>')  # Split the file into individual PDB entries
        # remove the first empty entry if it exists and add > to all entries
        if examples[0] == '':
            examples = examples[1:]
        examples = ['>' + example for example in examples]  # Ensure each
        return examples
    
def divide_pssm_to_N_parts(folder:str,pssm_file:str, num_parts:int):
    examples = divide_pssm_to_examples(pssm_file)
    with open(pssm_file, 'r') as f:
        all_file = f.read() 
        num_of_pdbs = all_file.count('>')  # Count the number of PDB entries

    indexes = np.linspace(0, num_of_pdbs, num_parts + 1, dtype=int) 
    print(f"Number of PDBs: {num_of_pdbs}, Indexes: {indexes}")
    for i in range(num_parts):
        start_index = indexes[i]
        end_index = indexes[i + 1]
        part_examples = examples[start_index:end_index]
        part_content = ''.join(part_examples)
        part_file_path = os.path.join(folder, f'part_{i + 1}.txt')
        with open(part_file_path, 'w') as part_file:
            part_file.write(part_content)

def unite_parts(scanNet_pssm_parts_folder:str,scanNet_asa_parts_folder:str,
                scanNet_not_included_parts_folder:str,num_parts:int,pssm_output_path:str,
                asa_output_path:str,not_included_output_path:str):
    scanNet_pssm_files = [os.path.join(scanNet_pssm_parts_folder, f'part_{i + 1}_pssm.txt') for i in range(num_parts)]
    scanNet_asa_files = [os.path.join(scanNet_asa_parts_folder, f'part_{i + 1}_asa.txt') for i in range(num_parts)]
    scanNet_not_included_files = [os.path.join(scanNet_not_included_parts_folder, f'part_{i + 1}_not_included.txt') for i in range(num_parts)]
    concat_files_with_newline_seperator(scanNet_pssm_files, pssm_output_path)
    concat_files_with_newline_seperator(scanNet_asa_files, asa_output_path)
    concat_files_with_newline_seperator(scanNet_not_included_files, not_included_output_path)


def sample_pssm_and_asa_files(input_pssm:str, input_asa:str, output_pssm_debug:str, output_asa_debug:str, sample_n:int, seed:int):



    """
    Create a smaller sample of the PSSM and ASA files by randomly sampling chain blocks.
    The same chain blocks (by index) are taken from both files.

    Args:
      input_pssm (str): path to original PSSM file
      input_asa (str): path to original ASA file
      output_pssm_debug (str): path to write sampled PSSM
      output_asa_debug (str): path to write sampled ASA
      sample_n (int): number of chain records to sample
      seed (int): random seed for reproducibility

    Returns:
      tuple: paths to the debug PSSM file and debug ASA file
    """
    def read_records(file_path):
        records = []
        with open(file_path, 'r') as f:
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
        return records

    pssm_records = read_records(input_pssm)
    asa_records = read_records(input_asa)

    if len(pssm_records) != len(asa_records):
        raise ValueError("The number of records in the PSSM and ASA files do not match.")

    random.seed(seed)
    if sample_n < len(pssm_records):
        indices = random.sample(range(len(pssm_records)), sample_n)
    else:
        indices = list(range(len(pssm_records)))

    with open(output_pssm_debug, 'w') as f_pssm, open(output_asa_debug, 'w') as f_asa:
        for i in indices:
            f_pssm.writelines(pssm_records[i])
            f_asa.writelines(asa_records[i])

    return output_pssm_debug, output_asa_debug


def filter_pssm_using_keys(pssm_file:str, keys:list, output_file:str):
    """
    Filter a PSSM file to only include records with specific keys.

    Args:
      pssm_file (str): path to the original PSSM file
      keys (list): list of keys to filter by (e.g., chain names)
      output_file (str): path to write the filtered PSSM file

    Returns:
      None
    """
    examples = divide_pssm_to_examples(pssm_file)
    filtered_examples = []
    for example in examples:
        key = example.split('\n')[0][1:]  # Get the key from the header line
        if key in keys:
            filtered_examples.append(example)
    with open(output_file, 'w') as f:
        chunk_size = 10000  # Write in chunks to avoid memory issues
        for i in range(0, len(filtered_examples), chunk_size):
            if i + chunk_size > len(filtered_examples):
                chunk_size = len(filtered_examples) - i
            chunk = filtered_examples[i:i + chunk_size]
            f.write(''.join(chunk))

def remove_non_valid_chains(pssm_file: str, asa_file: str, output_pssm_file: str, output_asa_file: str):
    """
    Remove chains that are not valid (i.e., have no labels or contain non valid amino acids)
    from the PSSM and ASA files.
    """
    asa_labels = parse_labels(asa_file)
    pssm_labels = parse_labels(pssm_file)
    valid_keys = [key[1:] for key in asa_labels.keys()]
    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    for key in asa_labels.keys():
        asa_content = asa_labels[key]
        pssm_content = pssm_labels[key]
        assert (len(asa_content) == len(pssm_content)), f"Length mismatch for key {key}: {len(asa_content)} vs {len(pssm_content)}"
        for i in range(len(asa_content)):
            if len(asa_content[i].split()) != 4 or len(pssm_content[i].split()) != 4:
                print(f"Invalid line in ASA or PSSM file for key {key} at index {i}: {asa_content[i]} | {pssm_content[i]}")
                if key[1:] in valid_keys:
                    valid_keys.remove(key[1:])
                break
        if len(asa_content) < 15 and key[1:] in valid_keys:
            print(f"Chain {key} has less than 15 amino acids, removing it.")
            valid_keys.remove(key[1:])
            continue
        # Check for non valid amino acids in the chain
        for line in asa_content:
            parts = line.split()
            if len(parts) < 3:
                continue
            aa = parts[2]
            if aa not in valid_aa:
                print(f"Chain {key} contains non valid amino acid '{aa}', removing it.")
                if key[1:] in valid_keys:
                    valid_keys.remove(key[1:])
                break
    filter_pssm_using_keys(pssm_file, valid_keys, output_pssm_file)
    filter_pssm_using_keys(asa_file, valid_keys, output_asa_file)
