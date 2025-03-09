import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import shutil
import paths
import pandas as pd
from results.ScanNet.uniprot_utils import get_chain_organism
from results.ScanNet.orientation_based_performence_analysis import look_for_class_in_string_util
import csv
from data_preparation.patch_to_score.protein_level_db_creation_utils import download_alphafold_model
from Bio.PDB.MMCIFParser import MMCIFParser
from data_preparation.ScanNet.db_creation_scanNet_utils import get_str_seq_of_chain


def find_class_for_chain(pdb_name, chain_name):
    class_dict_for_receptor = {'e1': False, 'e2': False, 'e3|e4': False, 'deubiquitylase': False}
    pdb_name_4_letters = pdb_name[:4]
    print((pdb_name_4_letters, chain_name))
    try:
        _, name, _, _, _ = get_chain_organism(pdb_name_4_letters, chain_name)
        look_for_class_in_string_util(name, class_dict_for_receptor)
    except:
        print('Exception! ')
    return class_dict_for_receptor

def update_csv_with_classes(input_csv, output_csv):
    with open(input_csv, mode='r') as infile, open(output_csv, mode='w', newline='') as outfile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames + ['e1', 'e2', 'e3|e4', 'deubiquitylase', 'other']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in reader:
            pdb_id = row['PDB_ID']
            chain_id = row['CHAIN_ID']
            class_dict = find_class_for_chain(pdb_id, chain_id)
            
            # Determine if 'other' should be true
            if not any(class_dict.get(key) for key in ['e1', 'e2', 'e3|e4', 'deubiquitylase']):
                class_dict['other'] = True
            else:
                class_dict['other'] = False
            
            row.update(class_dict)
            writer.writerow(row)

def download_all_uniprot_models_from_csv():
    # Define the path to the CSV file
    csv_path = os.path.join(paths.structural_aligners_path, 'pdb_chain_table_uniprot_with_classes.csv')
    
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_path)
    
    # Drop rows with NaN values in the 'uniprot' column
    df = df.dropna(subset=['uniprot'])
    
    # Iterate over the 'uniprot' column and download each model
    for uniprot_id in df['uniprot']:
        download_alphafold_model(uniprot_id,paths.pdb_files_structural_aligners_path)

def copy_pdb_files():
    source_dir = paths.structural_alignment_files_from_colab_path
    dest_dir = paths.pdb_files_structural_aligners_path
    
    for dir_name in os.listdir(source_dir):
        dir_path = os.path.join(source_dir, dir_name)
        if os.path.isdir(dir_path):
            for file_name in os.listdir(dir_path):
                if '_001_' in file_name and file_name.endswith('.pdb'):
                    src_file_path = os.path.join(dir_path, file_name)
                    dest_file_path = os.path.join(dest_dir, f"{dir_name}.pdb")
                    shutil.copy(src_file_path, dest_file_path)
                    break  # Assuming only one such file per directory

def create_fasta_file(uniprot_id, sequence):
    fasta_content = f">{uniprot_id}\n{sequence}"
    inputDir = paths.missing_uniprots_fasta_files_path
    filename = os.path.join(inputDir, f"{uniprot_id}.fasta")
    with open(filename, "w") as f:
        f.write(fasta_content)

def extract_sequence_from_cif_file(cif_file_path, chain_id):
    parser = MMCIFParser()
    structure = parser.get_structure('structure', cif_file_path)
    if chain_id == 'A-2':
        print('hi')
    for model in structure:
        for chain in model:
            if chain.id == chain_id:
                sequence = get_str_seq_of_chain(chain)
                return sequence
    return None

def create_missing_uniprots_fasta_files(pdb_chain_table_uniprot_file_path):
    df = pd.read_csv(pdb_chain_table_uniprot_file_path)
    
    for index, row in df.iterrows():
        pdb_id = row['PDB_ID']
        chain_id = row['CHAIN_ID']
        uniprot = row['uniprot']
        
        if pd.isna(uniprot):
            pdb_file_path = os.path.join(paths.chosen_assemblies_path, f'{pdb_id}.cif')
            sequence = extract_sequence_from_cif_file(pdb_file_path, chain_id)
            assert (sequence is not None)
            uniprot_id = f"{pdb_id}_{chain_id}"
            create_fasta_file(uniprot_id, sequence)

def replace_nan_with_pdb_chain(input_file: str):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(input_file)
    
    # Replace NaN values
    for index, row in df.iterrows():
        for col in df.columns:
            if pd.isna(row[col]):
                df.at[index, col] = f"{row['PDB_ID']}_{row['CHAIN_ID']}"
    
    # Save the modified DataFrame back to a CSV file
    df.to_csv(input_file, index=False)


if __name__ == '__main__':
    input_csv = os.path.join(paths.original_pdbs_with_augmentations_path,'pdb_chain_table_uniprot.csv')
    output_csv = os.path.join(paths.structural_aligners_path,'pdb_chain_table_uniprot_with_classes.csv')
    update_csv_with_classes(input_csv, output_csv)
    # download_all_uniprot_models_from_csv()
    # create_missing_uniprots_fasta_files(os.path.join(paths.scanNet_AF2_augmentations_path,'pdb_chain_table_uniprot.csv'))
    # copy_pdb_files()
    replace_nan_with_pdb_chain(os.path.join(paths.structural_aligners_path, 'pdb_chain_table_uniprot_with_classes.csv'))
    
