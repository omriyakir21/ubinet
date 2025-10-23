import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import shutil
import paths
import pandas as pd
from results.ScanNet.uniprot_utils import get_chain_organism
from results.ScanNet.orientation_based_performence_analysis import look_for_class_in_string_util
import csv
from data_preparation.patch_to_score.v0.protein_level_db_creation_utils import download_alphafold_model
from Bio.PDB.MMCIFParser import MMCIFParser
from data_preparation.ScanNet.db_creation_scanNet_utils import get_str_seq_of_chain,is_ubiquitin
from Bio import PDB
import subprocess


def find_class_for_chain(uniprot_id):
    class_dict_for_receptor = {'e1': False, 'e2': False, 'e3|e4': False, 'deubiquitylase': False}
    print(uniprot_id)
    try:
        _, description, _, _, _ = get_chain_organism(uniprot_id)
        look_for_class_in_string_util(description, class_dict_for_receptor)
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
            uniprot_id = row['uniprot']
            class_dict = find_class_for_chain(uniprot_id)
            
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

def create_substructures_for_all_chains(input_folder:str,chains_info_csv:str,folder_with_ubiqs:str,folder_without_ubiqs:str):
    chains_info = pd.read_csv(chains_info_csv)
    for _, row in chains_info.iterrows():
        pdb_id = row['PDB_ID']
        chain_id = row['CHAIN_ID']
        print(f"Processing {pdb_id} ,{chain_id}")
        if len(chain_id) > 1:
            print(f"Skipping {pdb_id} {chain_id}")
            continue
        input_path = os.path.join(input_folder, f"{pdb_id}.cif")
        output_path_with_ubiqs = os.path.join(folder_with_ubiqs, f"{pdb_id}_{chain_id}_with_ubiqs.pdb")
        extract_chains(input_path, chain_id, output_path_with_ubiqs, include_ubiquitin=True, input_type='cif')
        output_path_without_ubiqs = os.path.join(folder_without_ubiqs, f"{pdb_id}_{chain_id}_without_ubiqs.pdb")
        extract_chains(input_path, chain_id, output_path_without_ubiqs, include_ubiquitin=False, input_type='cif')

def extract_chains(input_path, chain_id, output_path, include_ubiquitin=False, input_type='pdb'):
    if input_type == 'pdb':
        parser = PDB.PDBParser(QUIET=True)
    elif input_type == 'cif':
        parser = PDB.MMCIFParser(QUIET=True)
    else:
        raise ValueError("Unsupported input type. Use 'pdb' or 'cif'.")

    structure = parser.get_structure('structure', input_path)
    
    io = PDB.PDBIO()
    io.set_structure(structure)
    
    class ChainSelect(PDB.Select):
        def __init__(self):
            self.existing_chain_ids = {chain.id for model in structure for chain in model}
        
        def accept_chain(self, chain):
            if include_ubiquitin:
                is_ubiq = is_ubiquitin(chain)
                print(f"chain id: {chain.id}, is_ubiquitin: {is_ubiq}")
                if chain.id == chain_id or is_ubiq:
                    if is_ubiq and len(chain.id) > 1:
                        new_chain_id = self.get_new_chain_id()
                        chain.id = new_chain_id
                    return True
            else:
                if chain.id == chain_id:
                    return True
            return False
        
        def get_new_chain_id(self):
            for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789":
                if char not in self.existing_chain_ids:
                    self.existing_chain_ids.add(char)
                    return char
            raise ValueError("Ran out of unique chain IDs")

    io.save(output_path, ChainSelect())

def create_db_list_for_foldseek(GO_folder_path: str, output_list_path: str):
    """
    Writes one absolute path per line (no headers, no tabs).
    Supports .pdb/.pdb.gz/.cif/.cif.gz.
    """
    subfolder_names = ['DUB', 'E1', 'E2', 'E3', 'ubiquitinBinding']
    exts = ('.pdb', '.pdb.gz', '.cif', '.cif.gz')
    n = 0
    with open(output_list_path, 'w', newline='\n') as out:
        for sub in subfolder_names:
            subdir = os.path.join(GO_folder_path, sub)
            if not os.path.isdir(subdir):
                print(f"Warning: {subdir} does not exist or is not a directory.")
                continue
            for fname in sorted(os.listdir(subdir)):
                if fname.lower().endswith(exts):
                    abs_path = os.path.abspath(os.path.join(subdir, fname))
                    out.write(abs_path + '\n')
                    n += 1
    if n == 0:
        raise RuntimeError(f"No structure files found under {GO_folder_path}")
    print(f"Wrote {n} paths to {output_list_path}")

if __name__ == '__main__':
    plan_dict = {'version':'v4',
                 'seq_id_threshold' : '0.95',
                 'update_csv_with_classes': False,
                 'replace_nan_with_pdb_chain': False,
                 'create_substructures_for_all_chains': False,
                 'create_foldseek_patch2score_positives_db':True
               }

    pdb_chain_table_path = f'datasets/scanNet/AF2_augmentations/pdbs_with_augmentations_{plan_dict["seq_id_threshold"]}/{plan_dict["version"]}/pdb_chain_table_0.95.csv'
    structural_aligners_tables_path = os.path.join(paths.structural_aligners_path,'tables')
    os.makedirs(structural_aligners_tables_path, exist_ok=True)
    tables_version_folder = os.path.join(structural_aligners_tables_path, plan_dict['version'])
    os.makedirs(tables_version_folder, exist_ok=True)
    pdb_chain_table_uniprot_path = os.path.join(tables_version_folder,'pdb_chain_uniprot_table.csv')
    # map_pdb_chains_to_uniprot(pdb_chain_table_path, pdb_chain_table_uniprot_path)
    pdb_chain_uniprot_classes_table = os.path.join(tables_version_folder,'pdb_chain_uniprot_with_classes_table.csv')
    structures_folder = os.path.join(paths.chosen_assemblies_path, plan_dict['version'])
    
    folder_with_ubiqs = os.path.join(paths.binding_chains_pdbs_with_ubiqs_path, plan_dict['version'])
    folder_without_ubiqs = os.path.join(paths.binding_chains_pdbs_without_ubiqs_path, plan_dict['version'])
    foldseek_path = os.path.join(paths.patch_to_score_dataset_path, 'foldseek')
    foldseek_db_paths_list = os.path.join(foldseek_path, 'positives_paths.tsv')
    foldseek_db_file = os.path.join(foldseek_path, 'positives_db')


    if plan_dict['update_csv_with_classes']:
        print('Updating csv with classes')
        update_csv_with_classes(pdb_chain_table_uniprot_path, pdb_chain_uniprot_classes_table)
    if plan_dict['replace_nan_with_pdb_chain']:
        print('Replacing nan with pdb_chain')
        replace_nan_with_pdb_chain(pdb_chain_uniprot_classes_table)
    if plan_dict['create_substructures_for_all_chains']:
        os.makedirs(folder_with_ubiqs, exist_ok=True)
        os.makedirs(folder_without_ubiqs, exist_ok=True)
        print('Creating substructures for all chains')
        create_substructures_for_all_chains(input_folder=structures_folder, chains_info_csv=pdb_chain_uniprot_classes_table,
                                           output_path_with_ubiqs=folder_with_ubiqs, output_path_without_ubiqs=folder_without_ubiqs)

    if plan_dict['create_foldseek_patch2score_positives_db']:
        print('Creating foldseek patch2score positives db')
        os.makedirs(foldseek_path, exist_ok=True)

        create_db_list_for_foldseek(
            GO_folder_path=paths.GO_source_patch_to_score_path,
            output_list_path=foldseek_db_paths_list
        )

        try:
            res = subprocess.run(
                ["foldseek", "createdb", foldseek_db_paths_list, foldseek_db_file],
                check=True, capture_output=True, text=True
            )
            print("STDOUT:\n", res.stdout)
            if res.stderr:
                print("STDERR:\n", res.stderr)
        except subprocess.CalledProcessError as e:
            print("STDOUT:\n", e.stdout)
            print("STDERR:\n", e.stderr)
            raise
                    

