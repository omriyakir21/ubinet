import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import paths
import csv
from data_preparation.ScanNet.LabelPropagationAlgorithm_utils import split_receptors_into_individual_chains
import shutil
from Bio.PDB.SASA import ShrakeRupley
from Bio.Blast import NCBIWWW
from Bio import SeqIO
import numpy as np
from create_tables_and_weights import cluster_sequences
import biotite.sequence as seq
import biotite.sequence.io.fasta as fasta
from biotite.application.blast import BlastWebApp
import logging
from Bio.PDB import MMCIFParser, MMCIFIO, Select,PDBParser,PDBIO
import pandas as pd
import csv
from Bio import SeqIO
from biotite.application.muscle import Muscle5App
from biotite.sequence import ProteinSequence
from biotite.sequence.io.fasta import FastaFile
from data_preparation.patch_to_score.data_development_utils import get_str_seq_of_chain
from data_preparation.ScanNet.db_creation_scanNet_utils import aa_out_of_chain,load_as_pickle,THREE_LETTERS_TO_SINGLE_AA_DICT
from data_preparation.ScanNet.LabelPropagationAlgorithm_utils import normalize_value,create_chain_dict_with_all_info_only_pssm
import time
import subprocess,fileinput
from data_preparation.patch_to_score.protein_level_db_creation_utils import download_alphafold_model
from collections import defaultdict


path2mmseqs = 'mmseqs'
path2mmseqstmp = '/specific/disk2/home/mol_group/tmp/'
path2mmseqsdatabases = '/specific/disk2/home/mol_group/sequence_database/MMSEQS/'

def call_mmseqs(
        input_file,
        output_file,
        database = 'SwissProt',
        nthreads = 6,
        filtermsa = True,
        cov = 0.0,
        qid = 0.0,
        maxseqid = 1.1,
        gapopen = 11,
        gapextend = 1,
        s = 5.7000,
        num_iterations = 1,
        maxseqs = 100000,
        overwrite = True,
        report=None

):
    t = time.time()
    if cov>1:
        cov = cov/100.
    if not overwrite:
        if os.path.exists(output_file):
            print('File %s already exists. Not recomputing' %output_file,file=report)
            return output_file

    ninputs = sum([line.startswith('>') for line in open(input_file,'r').readlines()])

    '''
    Source: https://github.com/soedinglab/MMseqs2/issues/693
    '''
    tmp_folder = '.'.join(output_file.split('.')[:-1]) + '/'
    os.makedirs(tmp_folder,exist_ok=True)
    tmp_input_file = os.path.join(tmp_folder,'input')
    tmp_output_file = os.path.join(tmp_folder, 'output')
    tmp_output_file2 = os.path.join(tmp_folder, 'output2')

    commands = [
        [path2mmseqs, 'createdb', input_file, tmp_input_file],
        [path2mmseqs, 'search', tmp_input_file, os.path.join(path2mmseqsdatabases,database), tmp_output_file, path2mmseqstmp,
        '-s',str(s),'--cov', str(cov),'--cov-mode','1','--diff',str(maxseqs), '--qid', str(qid), '--max-seq-id', str(maxseqid),'--gap-open', str(gapopen), '--gap-extend', str(gapextend), '--threads',str(nthreads),'--num-iterations',str(num_iterations),
           '--max-seqs',str(maxseqs)],
        [path2mmseqs, 'result2msa', tmp_input_file, os.path.join(path2mmseqsdatabases,database), tmp_output_file,tmp_output_file2,
        '--filter-msa',str(int(filtermsa)), '--cov',str(cov),'--diff',str(maxseqs),'--qid', str(qid), '--max-seq-id',str(maxseqid),'--msa-format-mode','5','--gap-open', str(gapopen), '--gap-extend',str(gapextend), '--threads',str(nthreads)],
        [path2mmseqs, 'convertalis', tmp_input_file, os.path.join(path2mmseqsdatabases,database), tmp_output_file,tmp_output_file +'.tab','--format-output',"target,theader"],
        [path2mmseqs, 'unpackdb', tmp_output_file2, tmp_folder, '--unpack-name-mode', '0', '--threads',str(nthreads)]
    ]
    for command in commands:
        print(' '.join(command))
        subprocess.call(command)
    table_labels = pd.read_csv(tmp_output_file +'.tab',sep='\t',header=None,index_col=0).drop_duplicates()

    for n in range(ninputs):
        if ninputs == 1:
            output_file_ = output_file
        else:
            output_file_ = output_file.split('.fasta')[0] + '_%s.fasta'%n
        os.rename(os.path.join(tmp_folder,str(n) ),output_file_)
        with fileinput.input(files=output_file_, inplace=True) as f:
            for line in f:
                if line.startswith('>'):
                    try:
                        newlabel = table_labels.loc[line[1:-1]].item()
                    except:
                        newlabel = 'na|%s|' % line[1:-1]
                    newline = '>' + newlabel + '\n'
                else:
                    newline = line
                print(newline, end='')
        assert os.path.exists(output_file_)
    subprocess.call(['rm', '-r', tmp_folder])
    print('Called mmseqs finished: Duration %.2f s' % (time.time() - t),file=report)
    return output_file

class ChainSelect(Select):
    def __init__(self, chain_id):
        self.chain_id = chain_id

    def accept_chain(self, chain):
        return chain.id == self.chain_id

class CuttedChainSelect(Select):
    def __init__(self, first_index, last_index,first_aa_index):
        self.first_index = first_index+first_aa_index
        self.last_index = last_index+first_aa_index
        print (f'in cutted chain select first_index = {first_index}, last_index = {last_index}')


    def accept_residue(self, residue):
        return self.first_index <= residue.get_id()[1] <= self.last_index

def process_pdb_chain_table(csv_path, structures_path, new_folder_path):
    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Initialize the MMCIF parser
    parser = MMCIFParser(QUIET=True)

    for index, row in df.iterrows():
        pdb_id = row['PDB_ID']
        chain_id = row['CHAIN_ID']
        
        # Create a new folder for each PDB_ID_CHAIN_ID
        new_subfolder = os.path.join(new_folder_path, f"{pdb_id}_{chain_id}")
        if not os.path.exists(new_subfolder):
            os.makedirs(new_subfolder)
        
        # Parse the structure file
        structure_file = os.path.join(structures_path, f"{pdb_id}.cif")
        structure = parser.get_structure(pdb_id, structure_file)
        
        # Select the chain and save the substructure as CIF
        io = MMCIFIO()
        io.set_structure(structure)
        io.save(os.path.join(new_subfolder, f"{pdb_id}_{chain_id}.cif"), ChainSelect(chain_id))


def get_str_seq_of_single_chain_protein(pdb_id, structure_file):
    parser = MMCIFParser()
    structure = parser.get_structure(pdb_id, structure_file)
    for model in structure:
        for chain in model:
            return get_str_seq_of_chain(chain)


def from_chains_dict_to_pdb_chain_table(chain_dict,augmentations_folder,seq_id):
    # Create the new folder path if it doesn't exist
    if not os.path.exists(augmentations_folder):
        os.makedirs(augmentations_folder)
    with open(os.path.join(augmentations_folder,f'pdb_chain_table_{seq_id}.csv'), mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header row
        writer.writerow(['PDB_ID', 'CHAIN_ID'])
        # Iterate over the chain_dict
        for pdb_chain, chains in chain_dict.items():
            pdb_id = pdb_chain.split('_')[0]
            for chain in chains:
                writer.writerow([pdb_id, chain])

def run_blast_online(query_sequence, db, output, evalue=1e-5,hitlist_size=50,entrez_query="(none)",matrix_name="BLOSUM62"):
    result_handle = NCBIWWW.qblast("blastp", db, query_sequence, expect=evalue, format_type="XML",
                                   hitlist_size=hitlist_size,entrez_query=entrez_query,matrix_name=matrix_name)
    with open(output, "w") as out_handle:
        out_handle.write(result_handle.read())



def find_all_close_homologs(input_folder,seq_id):
    cnt = 0
    homologs_set = set()
    sequences_set = set()
    for subfolder in os.listdir(input_folder):
        subfolder_path = os.path.join(input_folder, subfolder)
        if os.path.isdir(subfolder_path):
            pdb_file = os.path.join(subfolder_path, f"{subfolder}.cif")
            if os.path.exists(pdb_file):
                sequence = get_str_seq_of_single_chain_protein(subfolder, pdb_file)
                print(f'Processing {subfolder}, cnt: {cnt}')
                try:
                    find_close_homologs(subfolder, sequence, subfolder_path,seq_id,homologs_set,sequences_set)
                    logging.info(f"Processed {subfolder}, cnt: {cnt}")
                except Exception as e:
                    logging.error(f"Error processing {subfolder}: {e}", exc_info=True)
            cnt += 1

def create_fasta_file(sequence, id,output_folder):
    fasta_file = os.path.join(output_folder, f"{id}.fasta")
    with open(fasta_file, "w") as out_handle:
        out_handle.write(f">{id}\n{sequence}\n")
    return fasta_file


def find_close_homologs(id,sequence, output_folder,seq_id,homologs_set,sequences_set):
    logging.info(f"Finding close homologs for {id}, with sequence {sequence}")
    input_file = create_fasta_file(sequence, id, output_folder)
    output_file_colabfold = os.path.join(output_folder, f"{id}_homologs_colabfold.fasta")
    output_file_uniprot = os.path.join(output_folder, f"{id}_homologs_uniprot.fasta")
    output_csv_file = os.path.join(output_folder, f"{id}_homologs_representatives.csv")
    finished_file = os.path.join(output_folder, f"{id}_homologs_finished.txt")
    if os.path.exists(finished_file):
        logging.info(f"Homologs for {id} already found, skipping")
        return homologs_set
    call_mmseqs(
        input_file,
        output_file_colabfold,
        # database = 'UniProtKB',
        database = 'colabfold_envdb_202108_db',
        nthreads = 128,
        filtermsa = True,
        cov = 0.9,
        qid = seq_id,
        maxseqid = 1.1,
        gapopen = 11,
        gapextend = 1,
        s = 5.7000,
        num_iterations = 1,
        maxseqs = 100000,
        overwrite = True,
        report=None
    )

    call_mmseqs(
        input_file,
        output_file_uniprot,
        database = 'UniProtKB',
        nthreads = 128,
        filtermsa = True,
        cov = 0.9,
        qid = seq_id,
        maxseqid = 1.1,
        gapopen = 11,
        gapextend = 1,
        s = 5.7000,
        num_iterations = 1,
        maxseqs = 100000,
        overwrite = True,
        report=None
    )
    # Parse the sequences and their definitions
    all_sequences = []
    all_definitions = []
    all_uniprots = []
    cnt_sequences = 0
    with open(output_file_colabfold, "r") as f:
        lines_colabfold = f.readlines()
        for i in range(2, len(lines_colabfold), 2):  # Skip the first sequence
            if lines_colabfold[i].startswith('>'):
                definition = lines_colabfold[i].strip()
                hit_sequence = lines_colabfold[i + 1].strip().replace('-', '')
                if not hit_sequence.isupper():
                    continue
                if '|' in definition:
                    uniprot = definition.split('|')[1]
                else:
                    uniprot = definition.split(' ')[0]
                if uniprot.startswith('>'):
                    uniprot = uniprot[1:]
                if sequence != hit_sequence and sequence not in all_sequences and 'X' not in sequence:
                    all_definitions.append(definition)
                    all_sequences.append(hit_sequence)
                    all_uniprots.append(uniprot)
                    cnt_sequences += 1
        logging.info(f"cnt_sequences {cnt_sequences} found in the colabfoldDB search")
    cnt_sequences = 0
    with open(output_file_uniprot, "r") as f:
        lines_uniprot = f.readlines()
        for i in range(2, len(lines_uniprot), 2):  # Skip the first sequence
            if lines_uniprot[i].startswith('>'):
                definition = lines_uniprot[i].strip()
                hit_sequence = lines_uniprot[i + 1].strip().replace('-', '')
                if not hit_sequence.isupper():
                    continue
                if '|' in definition:
                    uniprot = definition.split('|')[1]
                else:
                    uniprot = definition.split(' ')[0]
                if uniprot.startswith('>'):
                    uniprot = uniprot[1:]
                if sequence != hit_sequence and sequence not in all_sequences and 'X' not in sequence:
                    all_definitions.append(definition)
                    all_sequences.append(hit_sequence)
                    all_uniprots.append(uniprot)
                    cnt_sequences += 1

        logging.info(f"cnt_sequences {cnt_sequences} found in the uniprotDB search")
    # Cluster the sequences and take cluster representatives
    cluster_indices, representative_indices = cluster_sequences(all_sequences,
                                                    seqid=0.98,coverage=0.9, covmode='0')
    print(f'cluster_indices {cluster_indices}')
    print(f'representative_indices {representative_indices}')
    # Count the size of each cluster
    cluster_sizes = {i: np.sum(cluster_indices == i) for i in set(cluster_indices)}
    print(f'cluster_sizes {cluster_sizes}')

    # Sort the clusters by size in descending order
    sorted_clusters = sorted(cluster_sizes.keys(), key=lambda x: cluster_sizes[x], reverse=True)

    # Take the top ten representatives that are not in homologs_set
    top_representatives = []
    for cluster in sorted_clusters:
        cluster_members = [i for i, x in enumerate(cluster_indices) if x == cluster]
        for idx in cluster_members:
            if homologs_set is None:
                raise Exception('homologos set is None')
            if all_uniprots[idx] not in homologs_set and 'X' not in all_sequences[idx] and all_sequences[idx] not in sequences_set:
                top_representatives.append(idx)
                homologs_set.add(all_uniprots[idx])
                sequences_set.add(all_sequences[idx])
                break
        if len(top_representatives) >= 10:
            break
    logging.info(f'homologos set size is {len(homologs_set)}')
    homologs = [all_sequences[i] for i in top_representatives]
    definitions = [all_definitions[i] for i in top_representatives]
    uniprots = [all_uniprots[i] for i in top_representatives]

    # Save the cluster representatives as the final paralogs

    with open(output_csv_file, "w") as csvfile:
        csvfile.write("sequence,uniprot,definition\n")
        for i in range(len(homologs)):
            homolog = homologs[i]
            definition = definitions[i]
            definition = definition.replace(',', '_')
            uniprot = uniprots[i]
            csvfile.write(f"{homolog},{uniprot},{definition}\n")

    # Create a finished file to indicate that the homologs were found
    with open(finished_file, "w") as f:
        f.write(f"Homologs for {id} were found and saved to {output_csv_file}\n")
        f.write(f"Total homologs found: {len(homologs_set)}\n")
        f.write(f"Total sequences processed: {len(all_sequences)}\n")
    
    return homologs_set

def clean_subfolders(folder_path):
    # Iterate through all subfolders in the given folder
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # Get the file extension
            file_extension = os.path.splitext(file)[1]
            # Check if the file extension is not .pdb or .xml
            if file_extension not in ['.pdb', '.xml']:
                # Construct the full file path
                file_path = os.path.join(root, file)
                # Remove the file
                os.remove(file_path)
                print(f"Removed: {file_path}")

def make_chain_dict_with_labels_info(chains_keys, chains_sequences, chains_labels, chain_names, lines, chains_asa_values):
    chain_dict = {}
    for i in range(len(chains_keys)):
        chain_dict[chain_names[i]] = {}
        chain_dict[chain_names[i]]['sequences'] = chains_sequences[i]
        chain_dict[chain_names[i]]['labels'] = chains_labels[i]
        chain_dict[chain_names[i]]['lines'] = lines[i]
        chain_dict[chain_names[i]]['asa_values'] = chains_asa_values[i]
    return chain_dict


def create_msas_from_folder(folder_path):
    # Extract pdb_id and chain_id from folder path
    folder_name = os.path.basename(folder_path)
    pdb_id, chain_id = folder_name.split('_')

    # Read the sequence from the structure file
    structure_file = os.path.join(folder_path, f"{pdb_id}_{chain_id}.cif")
    structure_seq = get_str_seq_of_single_chain_protein(pdb_id, structure_file)
    # Read homologs from the CSV file
    homologs_file = os.path.join(folder_path, f"{pdb_id}_{chain_id}_homologs_representatives.csv")
    sequences = [structure_seq]
    uniprots = []
    definitions = []
    with open(homologs_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for row in reader:
            sequences.append(row[0])
            uniprots.append(row[1])
    sequences = [ProteinSequence(sequence) for sequence in sequences]
    if len(sequences) == 1:
        msa_file = os.path.join(folder_path, f"{pdb_id}_{chain_id}_msa.txt")
        with open(msa_file, 'w') as msa_out:
            msa_out.write(f">{pdb_id}_{chain_id}\n")
            msa_out.write(structure_seq + '\n')
        sub_msa_file = os.path.join(folder_path, f"{pdb_id}_{chain_id}_sub_msa.txt")
        with open(sub_msa_file, 'w') as sub_msa_out:
            sub_msa_out.write(f">{pdb_id}_{chain_id}\n")
            sub_msa_out.write(structure_seq + '\n')
        return None 
        
    muscle_app = Muscle5App(sequences)
    muscle_app.start()
    muscle_app.join()
    alignment = muscle_app.get_alignment()
    gapped_seqs = alignment.get_gapped_sequences()

    # Save MSA with definitions and similarity
    msa_file = os.path.join(folder_path, f"{pdb_id}_{chain_id}_msa.txt")
    with open(msa_file, 'w') as msa_out:
        msa_out.write(f">{pdb_id}_{chain_id} \n")
        msa_out.write(str(gapped_seqs[0]) + '\n')
        for i, seq in enumerate(gapped_seqs[1:]):
            msa_out.write(f">{uniprots[i]}\n")
            msa_out.write(str(seq) + '\n')

    # Extract sub-MSA
    structure_msa_seq = str(gapped_seqs[0])
    first_non_gap = next(i for i, c in enumerate(structure_msa_seq) if c != '-')
    last_non_gap = len(structure_msa_seq) - next(i for i, c in enumerate(reversed(structure_msa_seq)) if c != '-')

    sub_msa_file = os.path.join(folder_path, f"{pdb_id}_{chain_id}_sub_msa.txt")
    with open(sub_msa_file, 'w') as sub_msa_out:
        sub_msa_out.write(f">{pdb_id}_{chain_id}\n")
        sub_msa_out.write(structure_msa_seq[first_non_gap:last_non_gap] + '\n')
        for i, seq in enumerate(gapped_seqs[1:]):
            sub_msa_out.write(f">{uniprots[i]}\n")
            sub_msa_out.write(str(seq)[first_non_gap:last_non_gap] + '\n')

def create_AF2_structures_for_folder(folder_path):
    folder_name = os.path.basename(folder_path)
    pdb_id, chain_id = folder_name.split('_')
    sub_msa_file = os.path.join(folder_path, f"{pdb_id}_{chain_id}_sub_msa.txt")
    with open(sub_msa_file, 'r') as file:
        lines = file.readlines()
        if len(lines) == 2:
            logging.info(f"Skipping {pdb_id}_{chain_id} as it has no homologs")
            return None
        sequences_file = os.path.join(folder_path, f"{pdb_id}_{chain_id}_homologs_cutted.fasta")
        with open(sequences_file, 'w') as seq_file:
            cnt = 0
            for i in range(3, len(lines), 2):
                line = lines[i]
                line = line.strip()
                sequence = line.split(' ')[0]
                sequence = sequence.replace('-', '')
                seq_file.write(f">{pdb_id}_{chain_id}_{cnt}\n")
                seq_file.write(sequence + '\n')
                cnt += 1

    

def create_msas_all_folders(parent_folder):
    cnt = 0
    for item in os.listdir(parent_folder):
        folder_path = os.path.join(parent_folder, item)
        if os.path.isdir(folder_path):
            try:
                logging.info(f"Processing {folder_path}, cnt={cnt}")
                create_msas_from_folder(folder_path)
                logging.info(f"Processed {folder_path}, cnt={cnt}")
            except Exception as e:
                logging.error(f"Error processing {folder_path}: {e}")   
            # create_AF2_structures_for_folder(folder_path)
            cnt += 1
        
def download_uniprots_for_folder(folder_path):
    folder_name = os.path.basename(folder_path)
    pdb_id, chain_id = folder_name.split('_')
    homologs_file = os.path.join(folder_path, f"{pdb_id}_{chain_id}_homologs_representatives.csv")
    with open(homologs_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for row in reader:
            uniprot_id = row[1]
            if not os.path.exists(os.path.join(folder_path, f"{uniprot_id}.pdb")):
                logging.info(f"Downloading {uniprot_id} for {folder_path}")
                download_alphafold_model(uniprot_id, folder_path)
def download_all_uniprots(parent_folder):
    cnt = 0
    for item in os.listdir(parent_folder):
        folder_path = os.path.join(parent_folder, item)
        if os.path.isdir(folder_path):
            try:
                logging.info(f" cnt={cnt}, downloading uniprots for {folder_path} ")
                download_uniprots_for_folder(folder_path)
                logging.info(f" cnt={cnt},Processed {folder_path}")
            except Exception as e:
                logging.error(f"Error processing {folder_path}: {e}")   
            # create_AF2_structures_for_folder(folder_path)
            cnt += 1

def calculate_ASAF(amino_acids,quantile_dict):
    sasa_calc = ShrakeRupley()
    asa_values = []
    for aa in amino_acids:
        sasa_calc.compute(aa, level="R")
        aa_char = THREE_LETTERS_TO_SINGLE_AA_DICT[aa.get_resname()]
        quantile5,quantile95 = quantile_dict[aa_char]
        asa_values.append(normalize_value(aa.sasa,quantile5,quantile95))
    return asa_values

def get_5_and_95_percentiles_for_asa_values(chains_asa_values):
    all_asa_values = np.concatenate(chains_asa_values)
    percentile_5 = np.percentile(all_asa_values, 5)
    percentile_95 = np.percentile(all_asa_values, 95)
    return percentile_5, percentile_95

        
def find_5_and_95_percentiles_for_folder(folder_path):
    folder_name = os.path.basename(folder_path)
    pdb_id, chain_id = folder_name.split('_')
    msa_file = os.path.join(folder_path, f"{pdb_id}_{chain_id}_msa.txt")
    with open(msa_file, 'r') as msa_out:
        lines = msa_out.readlines()
        asa_values = []
        structure_file = os.path.join(folder_path, f"{pdb_id}_{chain_id}.cif")
        parser = MMCIFParser()
        structure = parser.get_structure(pdb_id, structure_file)
        for model in structure:
            for chain in model:
                chain_aa = aa_out_of_chain(chain)
                asa_values.append(calculate_ASAF(chain_aa))
        for i in range(2, len(lines), 2):
            uniprot_id = lines[i][1:]
            uniprot_pdb_file = os.path.join(folder_path, f"{uniprot_id}.pdb")
            if os.path.exists(uniprot_pdb_file):
                parser = PDBParser()
                structure = parser.get_structure(uniprot_id, uniprot_pdb_file)
                for model in structure:
                    for chain in model:
                        chain_aa = aa_out_of_chain(chain)
                        asa_values.append(calculate_ASAF(chain_aa))
        percentile_5, percentile_95 = get_5_and_95_percentiles_for_asa_values(asa_values)
        return percentile_5, percentile_95


def find_initial_index_of_uniprot(chain_sequence, sequence):
    ungapped_seq = sequence.replace('-','')
    print(f'chain_sequence = {chain_sequence}')
    print(f'ungapped_seq = {ungapped_seq}')
    for i in range(len(chain_sequence)-len(ungapped_seq)+1):
        if chain_sequence[i:i+len(ungapped_seq)] == ungapped_seq:
            return i
    

def create_label_files_for_folder(folder_path,chain_dict,asa_thereshold,quantile_dict):
    folder_name = os.path.basename(folder_path)
    # percentile_5, percentile_95 = find_5_and_95_percentiles_for_folder(folder_path)
    pdb_id, chain_id = folder_name.split('_')
    msa_file = os.path.join(folder_path, f"{pdb_id}_{chain_id}_msa.txt")

    key = f"{pdb_id}_{chain_id}"
    # if key != '1nbf_A':
    #     return
    print(f'in {key}')
    original_sequence_normalized_asa_values = chain_dict[key]['asa_values']
    original_sequence_labels = chain_dict[key]['labels']
    write_lines = []
    if not os.path.exists(msa_file):
        logging.error(f"MSA file {msa_file} not found")
        return
    with open(msa_file, 'r') as msa_out:
        lines = msa_out.readlines()
        original_sequence = lines[1][:-1]
        first_non_gap = next(i for i, c in enumerate(original_sequence) if c != '-')
        last_non_gap = len(original_sequence) - next(i for i, c in enumerate(reversed(original_sequence)) if c != '-')
        for i in range(3, len(lines), 2):
            uniprot_id = lines[i-1][1:-1]
            print(f'uniprot_id = {uniprot_id}')
            sequence = lines[i][:-1].upper()
            uniprot_pdb_file = os.path.join(folder_path, f"{uniprot_id}.pdb")
            if os.path.exists(uniprot_pdb_file):
                parser = PDBParser()
                structure = parser.get_structure(uniprot_id, uniprot_pdb_file)
                for model in structure:
                    for chain in model:
                        chain_sequence = get_str_seq_of_chain(chain)
                        chain_aa = aa_out_of_chain(chain)                       
                        uniprot_aa_index = find_initial_index_of_uniprot(chain_sequence,sequence)
                        # print(f'uniprot_aa_index = {uniprot_aa_index}')
                        if uniprot_aa_index is None:
                            logging.error(f'blast sequence of {uniprot_id} not found in the chain sequence')
                            continue
                        original_sequence_index = 0
                        start_index = find_initial_index_of_uniprot(chain_sequence,sequence) -1
                        first_index_of_uniprot = start_index
                        normalized_asa_values = calculate_ASAF(chain_aa,quantile_dict)
                        chain_id = chain.id[0]
                        header = f">|{uniprot_id}|_{model.id}-{chain_id}\n"
                        write_lines.append(header)
                        last_index_of_uniprot = -1
                        for i in range(len(sequence)):
                            if i < first_non_gap:
                                if sequence[i] != '-':
                                    uniprot_aa_index += 1
                            elif i >= last_non_gap:
                                last_index_of_uniprot = uniprot_aa_index
                                break
                            else:
                                if sequence[i] != '-':
                                    if first_index_of_uniprot == start_index:
                                        first_index_of_uniprot = uniprot_aa_index
                                    if original_sequence[i] != '-':
                                        if normalized_asa_values[uniprot_aa_index] > min(asa_thereshold,0.75*float(original_sequence_normalized_asa_values[original_sequence_index])):
                                            label = original_sequence_labels[original_sequence_index]
                                        else:
                                            label = 0
                                        aa_index = str(chain_aa[uniprot_aa_index].get_id()[1])
                                        aa_letter = chain_sequence[uniprot_aa_index]
                                        original_sequence_index += 1
                                        line = f"{chain_id} {aa_index} {aa_letter} {label}\n"
                                        write_lines.append(line)
                                    uniprot_aa_index += 1
                                else:
                                    if original_sequence[i] != '-':
                                        original_sequence_index += 1
                        if last_index_of_uniprot == -1:
                            last_index_of_uniprot =  uniprot_aa_index - 1
                        last_index_of_uniprot = min(last_index_of_uniprot, len(chain_aa)-1)
                        io = PDBIO()
                        io.set_structure(structure)
                        io.save(os.path.join(folder_path, f"{uniprot_id}_cutted.pdb"), CuttedChainSelect(first_index_of_uniprot,last_index_of_uniprot,chain_aa[0].get_id()[1]))
    augmentations_file = os.path.join(folder_path, f"{pdb_id}_{chain_id}_augmentations_asaThreshold_{asa_thereshold}.txt")

    with open(augmentations_file, 'w') as aug_file:
        aug_file.writelines(write_lines)

    with open(augmentations_file, 'r') as aug_file:
        cnt = sum(1 for line in aug_file if line.startswith('>'))
    logging.info(f"Number of augmentations: {cnt}")
    return cnt

                                   
def make_chain_dict(chain_names):
    chainDict = dict()
    for chainName in chain_names:
        receptorName = chainName.split('$')[0]
        chainId = chainName.split('$')[1]
        if receptorName in chainDict:
            chainDict[receptorName].append(chainId)
        else:
            chainDict[receptorName] = []
            chainDict[receptorName].append(chainId)
    return chainDict




def create_label_files_for_augmentations(parent_folder,chain_dict,asa_thereshold,quantile_dict):
    cnt = 0
    all_augmentations = 0
    for item in os.listdir(parent_folder):
        folder_path = os.path.join(parent_folder, item)
        if os.path.isdir(folder_path):
            try:
                logging.info(f"Processing {folder_path}, cnt={cnt}")
                new_augmentations = create_label_files_for_folder(folder_path,chain_dict,asa_thereshold,quantile_dict)
                all_augmentations += new_augmentations
                logging.info(f"Processed {folder_path}, added {new_augmentations} augmentations, cnt={cnt}")
            except Exception as e:
                logging.error(f"Error processing {folder_path}: {e}")
            cnt += 1
    logging.info(f"Total number of augmentations: {all_augmentations}")




def create_chain_dict_with_all_info(chains_keys, chains_sequences, chains_labels, chain_names, lines, chains_asa_values, chains_asa_not_normalized_values):
    chain_dict = {}
    for i in range(len(chain_names)):
        id  =chain_names[i]
        id = id.split('_')[0] + '_' + id.split('_')[1].split('$')[-1]
        chain_dict[id] = {}
        chain_dict[id]['sequences'] = chains_sequences[i]
        chain_dict[id]['labels'] = chains_labels[i]
        chain_dict[id]['lines'] = lines[i]
        chain_dict[id]['asa_values'] = chains_asa_values[i]
        chain_dict[id]['asa_not_normalized_values'] = chains_asa_not_normalized_values[i]
    return chain_dict


def create_fasta_homologs_folder(input_folder, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Walk through all subfolders in the input folder
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith("homologs_representatives.csv"):
                file_path = os.path.join(root, file)
                
                # Open and parse the CSV file
                with open(file_path, mode='r') as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        sequence = row['sequence']
                        uniprot = row['uniprot']
                        
                        # Create FASTA content
                        fasta_content = f">{uniprot}\n{sequence}\n"
                        
                        # Define the output file path
                        output_file_path = os.path.join(output_folder, f"{uniprot}.fasta")
                        
                        # Write the FASTA file
                        with open(output_file_path, mode='w') as fasta_file:
                            fasta_file.write(fasta_content)


def fetch_AF2_structures(folder_path, AF2_structures_path):
    cnt = 0
    for subdir, _, files in os.walk(folder_path):
        logging.info(f"Processing {subdir}, cnt={cnt}")
        for file in files:
            if file.endswith('homologs_representatives.csv'):
                csv_path = os.path.join(subdir, file)
                try:
                    df = pd.read_csv(csv_path)
                except Exception as e:
                    logging.error(f"Error reading {csv_path}: {e}")
                    continue
                uniprot_ids = df['uniprot'].tolist()
                
                for uniprot_id in uniprot_ids:
                    found = False
                    for af2_file in os.listdir(AF2_structures_path):
                        if af2_file.startswith(f"{uniprot_id}_unrelaxed_rank_001") and af2_file.endswith(".pdb"):
                            source_file = os.path.join(AF2_structures_path, af2_file)
                            destination_file = os.path.join(subdir, f"{uniprot_id}.pdb")
                            shutil.copy(source_file, destination_file)
                            found = True
                            break
                    if not found:
                        print(f"No matching file found for {uniprot_id} in {AF2_structures_path}")
        cnt += 1

def parse_augmentation_file(file_path):
    augmentations = defaultdict(dict)
    with open(file_path, 'r') as f:
        current_key = None
        for line in f:
            if line.startswith('>'):
                current_key = line.strip()
                augmentations[current_key] = {}
            else:
                parts = line.split()
                index = int(parts[1])
                chain = parts[0]
                aa_type = parts[2]
                label = int(parts[3])
                augmentations[current_key][index] = (chain, aa_type, label)
    return augmentations

def merge_augmentations(augmentations_list):
    merged_augmentations = defaultdict(dict)
    for key in augmentations_list[0].keys():
        indices = set(augmentations_list[0][key].keys())
        for augmentations in augmentations_list[1:]:
            indices &= set(augmentations[key].keys())
        for index in indices:
            max_label = max(augmentations[key][index][2] for augmentations in augmentations_list)
            chain, aa_type, _ = augmentations_list[0][key][index]
            merged_augmentations[key][index] = (chain, aa_type, max_label)
    return merged_augmentations

def write_augmentation_file(file_path, augmentations):
    with open(file_path, 'w') as f:
        for key, values in augmentations.items():
            f.write(f"{key}\n")
            for index, (chain, aa_type, label) in sorted(values.items()):
                f.write(f"{chain} {index} {aa_type} {label}\n")

def create_pdb_chain_to_fold_dict(folder,prefix):
    pdb_dict = {}
    for i in range(5):
        file_path = os.path.join(folder, f'{prefix}{i}.txt')
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for j in range(len(lines)) :
                line = lines[j]
                if line.startswith('>'):
                    pdb_chains = []
                    pdb_name = line[1:5]
                else:
                    chain = line.split()[0]
                    if chain not in pdb_chains:
                        pdb_name_and_chain = f"{pdb_name}_{chain}"
                        pdb_dict[pdb_name_and_chain] = i
                        pdb_chains.append(chain)
    return pdb_dict

def merge_pssm_with_augmentations(PSSM_folder,augmentations_folder,file_prefix,ASA_THRESHOLD_VALUE):
    
    pdb_to_fold_dict = create_pdb_chain_to_fold_dict(PSSM_folder,file_prefix) 
    output_files = [open(os.path.join(PSSM_folder, f'{prefix}{i}_with_augmentations.txt'), 'w') for i in range(5)]
    number_of_augmentations = 0
    try:
        for i in range(5):
            pssm_file_path = os.path.join(PSSM_folder, f'{prefix}{i}.txt')
            with open(pssm_file_path, 'r') as pssm_file:
                output_files[i].write(pssm_file.read())
        
        for subfolder in os.listdir(augmentations_folder):
            subfolder_path = os.path.join(augmentations_folder, subfolder)
            if os.path.isdir(subfolder_path):
                fold_num = pdb_to_fold_dict.get(subfolder)
                if fold_num is None:
                    logging.error(f"Could not find fold number for {subfolder}")
                else:
                    augmentations_file_path = 'not found'
                    for file in os.listdir(subfolder_path):
                        if file.endswith(f'augmentations_asaThreshold_{ASA_THRESHOLD_VALUE}.txt'):
                            augmentations_file_path = os.path.join(subfolder_path, file)
                            break
                    if os.path.exists(augmentations_file_path):
                        with open(augmentations_file_path, 'r') as aug_file:
                            cnt = sum(1 for line in aug_file if line.startswith('>'))
                            number_of_augmentations += cnt
                    
                        with open(augmentations_file_path, 'r') as aug_file:
                            output_files[fold_num].write(aug_file.read())
    finally:
        for file in output_files:
            file.close()
    logging.info(f"Number of augmentations: {number_of_augmentations}")

def create_cutted_augmentations_folder(augmentations_folder):
    # Create the new folder
    cutted_augmentations_folder = os.path.join(augmentations_folder, 'cutted_augmentations')
    os.makedirs(cutted_augmentations_folder, exist_ok=True)
    print(f"Created folder: {cutted_augmentations_folder}")
    
    # Iterate over all subfolders in augmentations_folder
    for root, dirs, files in os.walk(augmentations_folder):
        # Skip the new folder to avoid infinite recursion
        if root == cutted_augmentations_folder:
            continue
        
        for file in files:
            if file.endswith('cutted.pdb'):
                # Copy the file to the new folder
                src_file = os.path.join(root, file)
                dst_file = os.path.join(cutted_augmentations_folder, file)
                shutil.copy2(src_file, dst_file)
    

def divide_msa_folder(msa_folder, output_folder):

    # Create 5 subfolders: msa_folder_0 to msa_folder_4
    subfolders = []
    for i in range(5):
        subfolder_path = os.path.join(output_folder, f"msa_folder_{i}")
        os.makedirs(subfolder_path, exist_ok=True)
        subfolders.append(subfolder_path)

    # List all files in msa_folder (only regular files)
    files = [file for file in os.listdir(msa_folder)
             if os.path.isfile(os.path.join(msa_folder, file))]
    
    # Distribute files into the 5 folders in a round-robin manner
    for index, file in enumerate(files):
        src_file = os.path.join(msa_folder, file)
        dest_subfolder = subfolders[index % 5]
        dest_file = os.path.join(dest_subfolder, file)
        shutil.copy2(src_file, dest_file)
    
    print(f"Files divided into: {', '.join(subfolders)}")

def integrate_predictions(input_folder, predictions_folder):
    """
    Integrate prediction files from 5 subfolders (AF2_predictions_0 to AF2_predictions_4)
    within input_folder into a single predictions_folder.
    """
    import os
    import shutil

    os.makedirs(predictions_folder, exist_ok=True)

    # Iterate through the 5 expected subfolders
    for i in range(5):
        subfolder = os.path.join(input_folder, f"AF2_predictions_{i}")
        if not os.path.exists(subfolder):
            print(f"Subfolder {subfolder} does not exist, skipping.")
            continue

        for file in os.listdir(subfolder):
            src_file = os.path.join(subfolder, file)
            if os.path.isfile(src_file):
                dst_file = os.path.join(predictions_folder, file)
                shutil.copy2(src_file, dst_file)

    print(f"Integrated predictions from 5 subfolders into {predictions_folder}")

if __name__ == '__main__':
    plan_dict = {
        'name' : 'v2',
        'seq_id' : '0.95',
        'ASA_THRESHOLD_VALUE' : 0.1,
        'weight_bound_for_ubiq_fold':0.21,
        'create_pdb_chain_table' : False,
        'find_homologs' : False,
        'create_fasta':False,
        'divide_msa': False,
        'integrate_predictions': False,
        'fetch_AF2_structures': False,
        'create_label_files': False,
        'merge_pssm_with_augmentations': True,
        'create_cutted_augmentations_folder': True,
        }
    
    
    
    with_scanNet = True
    with_scanNet_addition = '_with_scanNet_' if with_scanNet else ''
    augmentations_path = os.path.join(paths.pdbs_with_augmentations_95_path,plan_dict['name']) if plan_dict['seq_id'] == '0.95' else \
        os.path.join(paths.pdbs_with_augmentations_90_path,plan_dict['name'])
    os.makedirs(augmentations_path, exist_ok=True)
    ASA_path = os.path.join(paths.ASA_path,plan_dict['name'])
    PSSM_path = os.path.join(paths.PSSM_path, plan_dict['name'])
    PSSM_seq_id_folder = os.path.join(PSSM_path,f'seq_id_{plan_dict["seq_id"]}_asaThreshold_{plan_dict["ASA_THRESHOLD_VALUE"]}')
    chosen_assemblies_path = os.path.join(paths.chosen_assemblies_path,plan_dict['name'])
    pssm_content_path = os.path.join(PSSM_seq_id_folder, f'propagatedPssmFile_only_ubiq_{plan_dict["seq_id"]}_asaThreshold_{plan_dict["ASA_THRESHOLD_VALUE"]}.txt')
    asa_content_path = os.path.join(ASA_path, f'normalizedFullASAPssmContent.txt')
    asa_not_normalized_content_path = os.path.join(ASA_path, 'Integrated_Checkchains_asa_mer_filtered.txt')
    
    if plan_dict['create_pdb_chain_table']:
        chains_keys, chains_sequences, chains_labels, chain_names, lines, chains_asa_values= split_receptors_into_individual_chains(
            pssm_content_path, asa_content_path)
        _,_,_,_,_,chains_asa_not_normlized_values = split_receptors_into_individual_chains(pssm_content_path,asa_not_normalized_content_path)
        chain_dict = create_chain_dict_with_all_info(chains_keys, chains_sequences, chains_labels, chain_names, lines, chains_asa_values, chains_asa_not_normlized_values)
        chain_names_dict = make_chain_dict(chain_names)
        from_chains_dict_to_pdb_chain_table(chain_names_dict,augmentations_path,plan_dict['seq_id'])
        process_pdb_chain_table(os.path.join(augmentations_path,f'pdb_chain_table_{plan_dict["seq_id"]}.csv'), chosen_assemblies_path, augmentations_path)
    
    # # # find homologs 
    scanNet_AF2_augmentations_path = os.path.join(paths.scanNet_AF2_augmentations_path,plan_dict['name'])
    os.makedirs(scanNet_AF2_augmentations_path, exist_ok=True)

    if plan_dict['find_homologs']:
        log_file = os.path.join(scanNet_AF2_augmentations_path, f"AF2_augmentations_{plan_dict['seq_id']}.log")
        logging.basicConfig(filename=log_file, level=logging.INFO, 
                            format='%(asctime)s %(levelname)s:%(message)s')
        find_all_close_homologs(augmentations_path,float(plan_dict['seq_id'])) 
    
    if plan_dict['create_fasta']:
        create_fasta_homologs_folder(augmentations_path,os.path.join(augmentations_path,'fasta_folder'))

    msa_folder = os.path.join(augmentations_path, 'MSA_folder')
    msa_parts_folder = os.path.join(augmentations_path, 'MSA_parts_folder')
    os.makedirs(msa_parts_folder, exist_ok=True)
    if plan_dict['divide_msa']:
        divide_msa_folder(msa_folder, msa_parts_folder)
    af2_predictions_path = os.path.join(augmentations_path,'AF2_predictions')
    os.makedirs(af2_predictions_path, exist_ok=True)
    af2_predictions_parts_path = os.path.join(augmentations_path, 'AF2_predictions_parts')
    os.makedirs(af2_predictions_parts_path, exist_ok=True)
    if plan_dict['integrate_predictions']:
        integrate_predictions(af2_predictions_parts_path, af2_predictions_path)
    if plan_dict['fetch_AF2_structures']:
        fetch_AF2_structures(augmentations_path, af2_predictions_path)
    if plan_dict['create_label_files']:
        helper_path = os.path.join(paths.data_preperation_helper_path, plan_dict['name'])
        quantile_amino_acids_dict_path = os.path.join(helper_path, 'quantile_amino_acids_dict.pkl')
        quantile_dict = load_as_pickle(quantile_amino_acids_dict_path)
        chains_keys, chains_sequences, chains_labels, chain_names, lines, chains_asa_values= split_receptors_into_individual_chains(
            pssm_content_path, asa_content_path)
        _,_,_,_,_,chains_asa_not_normlized_values = split_receptors_into_individual_chains(pssm_content_path,asa_not_normalized_content_path)
        chain_dict = create_chain_dict_with_all_info(chains_keys, chains_sequences, chains_labels, chain_names, lines, chains_asa_values, chains_asa_not_normlized_values)
        create_msas_all_folders(augmentations_path)
        create_label_files_for_augmentations(augmentations_path,chain_dict,plan_dict['ASA_THRESHOLD_VALUE'],quantile_dict)
   
    if plan_dict['merge_pssm_with_augmentations']:
        bound_addition = f"_{str(plan_dict['weight_bound_for_ubiq_fold']).split('.')[1]}"
        prefix = f'PSSM{bound_addition}_'
        merge_pssm_with_augmentations(PSSM_seq_id_folder, augmentations_path,prefix,plan_dict['ASA_THRESHOLD_VALUE'])
    
    if plan_dict['create_cutted_augmentations_folder']:
        create_cutted_augmentations_folder(augmentations_path)