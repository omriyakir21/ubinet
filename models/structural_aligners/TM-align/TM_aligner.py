import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..','..'))
import shutil
import paths
import pandas as pd
import subprocess
import csv
from tmtools import tm_align
from Bio import PDB
import numpy as np
from data_preparation.ScanNet.db_creation_scanNet_utils import get_str_seq_of_chain
import ast
from data_preparation.patch_to_score.protein_level_db_creation_utils import download_alphafold_model
from numpy.linalg import inv
from Bio.PDB import MMCIFParser, PDBIO

def extract_coordinates(pdb_file):
    """
    Extracts the coordinates of C-alpha atoms from a PDB file and returns them as a numpy array.
    
    Parameters:
    pdb_file (str): Path to the PDB file.
    
    Returns:
    np.array: Array of C-alpha atomic coordinates.
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)
    
    coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if PDB.is_aa(residue):  # Ensure it's an amino acid
                    if 'CA' in residue:  # Check if C-alpha atom exists
                        ca_atom = residue['CA']  # Get the C-alpha atom
                        coords.append(ca_atom.get_coord())
    
    return np.array(coords)

def extract_sequence(pdb_file):
    """
    Extracts the sequence from a PDB file and returns it as a string.
    
    Parameters:
    pdb_file (str): Path to the PDB file.
    
    Returns:
    str: Amino acid sequence.
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)
    for model in structure:
        for chain in model:
            seq = get_str_seq_of_chain(chain)
    return seq



def align_uniprots_with_target(target_pdb, pdb_chain_table_path, output_dir_path):
    # Read the pdb_chain_table_uniprot_with_classes file
    with open(pdb_chain_table_path, 'r') as file:
        reader = csv.DictReader(file)
        rows = list(reader)
    target_coord = extract_coordinates(target_pdb)
    target_seq = extract_sequence(target_pdb)
    # Define the classes
    classes = ['e1', 'e2', 'e3|e4', 'deubiquitylase']
    
    # Dictionary to store the top TM score, corresponding uniprot, and alignment for each class
    top_scores = {class_name: {'tm_score': 0.0, 'uniprot': None, 'correspondence': None} for class_name in classes}
    
    # Iterate over each class
    for class_name in classes:
        # Filter rows where the class column is True
        filtered_rows = [row for row in rows if row[class_name] == 'True']
        
        # Align each uniprot with the target_pdb and find the top TM score
        for row in filtered_rows:
            uniprot_id = row['uniprot']
            uniprot_path = os.path.join(paths.pdb_files_structural_aligners_path, f'{uniprot_id}.pdb')
            if not os.path.exists(uniprot_path):
                continue
            ref_coord = extract_coordinates(uniprot_path)
            ref_seq = extract_sequence(uniprot_path)
            # Perform the alignment
            print(f'len target_seq: {len(target_seq)}')
            print(f'len ref_seq: {len(ref_seq)}')
            print(f'shape target_coord: {target_coord.shape}')
            print(f'shape ref_coord: {ref_coord.shape}')
            result = tm_align(target_coord, ref_coord, target_seq, ref_seq)
            tm_score = result.tm_norm_chain2  # Adjust based on result format
            translation = result.t
            rotation = result.u
            rmsd = result.rmsd
            
            # Update the top score if the current score is higher
            if tm_score > top_scores[class_name]['tm_score']:
                top_scores[class_name]['tm_score'] = tm_score
                top_scores[class_name]['uniprot'] = uniprot_id
                top_scores[class_name]['translation'] = translation
                top_scores[class_name]['rotation'] = rotation
                top_scores[class_name]['rmsd'] = rmsd    
    # Save the top scores and correspondence to a CSV file
    output_csv_path = os.path.join(output_dir_path,f'{os.path.splitext(os.path.basename(target_pdb))[0]}_top_scores.csv')
    with open(output_csv_path, 'w', newline='') as csvfile:
        fieldnames = ['class', 'tm_score', 'uniprot', 'translation', 'rotation', 'rmsd']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for class_name, score_data in top_scores.items():
            writer.writerow({
                'class': class_name,
                'tm_score': score_data['tm_score'],
                'uniprot': score_data['uniprot'],
                'translation': np.array2string(score_data['translation'], separator=','),
                'rotation': np.array2string(score_data['rotation'], separator=','),
                'rmsd': score_data['rmsd']
            })
    
    return top_scores

def parse_top_scores(csv_file_path):
    top_scores = {}
    
    with open(csv_file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        
        for row in reader:
            class_name = row['class']
            top_scores[class_name] = {
                'tm_score': float(row['tm_score']),
                'uniprot': row['uniprot'],
                'translation': np.array(ast.literal_eval(row['translation'])),
                'rotation': np.array(ast.literal_eval(row['rotation'])),
                'rmsd': float(row['rmsd'])
            }
    
    return top_scores

def parse_pdb_file(pdb_path):
    """
    :param pdb_path:
    :return: biopython's structure object
    """
    parser = PDB.PDBParser()
    strct_name = pdb_path.split('.')[0]
    structure = parser.get_structure(strct_name, pdb_path)
    return structure


def write_pdb_file(structure, output_path):
    """
    :param structure: biopython's structure object
    :param output_path:
    """
    io = PDB.PDBIO()
    io.set_structure(structure)
    io.save(output_path)
    return output_path


def apply_transformation(structure, matrix):
    """
    Transforms structure based on the transformation matrix
    :param structure: biopython's structure object
    :param matrix: transformation matrix dict
    :return: transformed structure
    """
    # rotation = np.asmatrix(np.array(matrix['rotation']))
    rotation = matrix['rotation']
    # translation = np.array(matrix['translation'])
    translation = matrix['translation']

    for atom in structure.get_atoms():
        atom.transform(rotation,translation)
    # apply transformation to each atom
    # map(lambda atom: atom.transform(rotation, translation), structure.get_atoms())

    return structure

# credit to joannalange / pdb-transform github for the transformation code
def pdb_transform(pdb_path, matrix, output_path):
    """
    Applies transformation matrix to the PDB file and writes the new PDB to file
    :param pdb_path: path to the input PDB file
    :param matrix:
        {
        'rotation': [
            x rotation vector (list),
            y rotation vector (list),
            z rotation vector (list)
        ],
        'translation': translation vector (list)
        }
    :param output_path: path to the output PDB file
    """
    structure = parse_pdb_file(pdb_path)

    structure = apply_transformation(structure, matrix)

    write_pdb_file(structure, output_path)

def align_transform(mov_pdb,ref_pdb,dir_path):
    mov_coord = extract_coordinates(mov_pdb)
    mov_seq = extract_sequence(mov_pdb)
    ref_coord = extract_coordinates(ref_pdb)
    ref_seq = extract_sequence(ref_pdb)
    result = tm_align(mov_coord, ref_coord, mov_seq, ref_seq)
    tm_score = result.tm_norm_chain2  # Adjust based on result format
    translation = result.t
    rotation = inv(result.u)
    rmsd = result.rmsd
    print(f'tm_score: {tm_score}')
    print(f'translation: {translation}')
    print(f'rotation: {rotation}')
    print(f'rmsd: {rmsd}')
    matrix = {}
    matrix['rotation'] = rotation
    matrix['translation'] = translation
    output_dir = os.path.join(dir_path, f'{os.path.splitext(os.path.basename(mov_pdb))[0]}_to_{os.path.splitext(os.path.basename(ref_pdb))[0]}')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    shutil.copy(ref_pdb, output_dir)
    output_path = os.path.join(output_dir,f'{os.path.splitext(os.path.basename(mov_pdb))[0]}_to_{os.path.splitext(os.path.basename(ref_pdb))[0]}.pdb')
    pdb_transform(mov_pdb, matrix, output_path)
    return tm_score, matrix

def convert_cif_to_pdb(input_cif_path, output_pdb_path):
    # Create parser and structure object
    parser = MMCIFParser()
    structure = parser.get_structure('structure', input_cif_path)
    
    # Create PDBIO object and save the structure to a PDB file
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_pdb_path)

def align_uniprot_with_most_similiar_from_class(uniprot,class_ubiq,dir_path):
    target_pdb =os.path.join(paths.AFDB_source_patch_to_score_path,'Human',f'{uniprot}.pdb') 
    top_scores = parse_top_scores(os.path.join(output_dir_path,f'{os.path.splitext(os.path.basename(target_pdb))[0]}_top_scores.csv'))
    matrix = {}
    matrix['rotation'] = top_scores[class_ubiq]['rotation']
    matrix['translation'] = top_scores[class_ubiq]['translation']

    ref_pdb = os.path.join(paths.pdb_files_structural_aligners_path, f'{top_scores[class_ubiq]["uniprot"]}.pdb')
    output_dir = os.path.join(dir_path, f'{uniprot}_to_{top_scores[class_ubiq]["uniprot"]}_{class_ubiq}')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    shutil.copy(ref_pdb, output_dir)
    output_path = os.path.join(output_dir,f'{uniprot}_to_{top_scores[class_ubiq]["uniprot"]}.pdb')
    pdb_transform(target_pdb, matrix, output_path)

if __name__ == '__main__':

    pdb_chain_table_path = os.path.join(paths.structural_aligners_path,'pdb_chain_table_uniprot_with_classes.csv')
    output_dir_path = paths.TM_aligner_aligned_pdbs_path
    
    # uniprots = ['O95793','Q8IYS0','Q86VN1','Q9NQL2','P07900','P53061']
    # for uniprot in uniprots:
    #     target_pdb =os.path.join(paths.AFDB_source_patch_to_score_path,'Human',f'{uniprot}.pdb') 
    #     top_scores = align_uniprots_with_target(target_pdb, pdb_chain_table_path,output_dir_path)
    align_uniprot_with_most_similiar_from_class('Q8IYS0','e2',paths.TM_aligner_transformed_pdbs_path)
    
    # pdb_transform(target_pdb, matrix, output_path)
    # # download_alphafold_model('P02185','/home/iscb/wolfson/omriyakir/ubinet/models/structural_aligners/TM-align/aligment_examples')
    # download_alphafold_model('P69905','/home/iscb/wolfson/omriyakir/ubinet/models/structural_aligners/TM-align/aligment_examples')
    # mov_exmpale_pdb = '/home/iscb/wolfson/omriyakir/ubinet/models/structural_aligners/TM-align/aligment_examples/P02185.pdb'
    # ref_example_pdb = '/home/iscb/wolfson/omriyakir/ubinet/models/structural_aligners/TM-align/aligment_examples/P69905.pdb'
    # align_transform(mov_exmpale_pdb,ref_example_pdb,paths.TM_aligner_transformed_pdbs_path)
 