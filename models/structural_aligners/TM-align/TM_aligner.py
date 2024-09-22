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
from tmtools.io import get_residue_data, get_structure
from tmtools.helpers import transform_structure
from tmtools.testing import get_pdb_path
from tmtools import tm_align


def align_uniprots_with_target(target_pdb, pdb_chain_table_path, output_dir_path):
    mobile_struct = get_structure(target_pdb)
    mobile_coords, mobile_seq = get_residue_data(next(mobile_struct.get_chains()))
    mobile_name = os.path.splitext(os.path.basename(target_pdb))[0]
    with open(pdb_chain_table_path, 'r') as file:
        reader = csv.DictReader(file)
        rows = list(reader)
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
            guide_struct = get_structure(uniprot_path)
            guide_coords, guide_seq = get_residue_data(next(guide_struct.get_chains()))
            res = tm_align(mobile_coords, guide_coords, mobile_seq, guide_seq)

            tm_score = res.tm_norm_chain2  # Adjust based on result format
            rmsd = res.rmsd
            
            # Update the top score if the current score is higher
            if tm_score > top_scores[class_name]['tm_score']:
                top_scores[class_name]['tm_score'] = tm_score
                top_scores[class_name]['uniprot'] = uniprot_id
                top_scores[class_name]['res'] = res
                top_scores[class_name]['rmsd'] = rmsd    
    
    mobile_folder = os.path.join(output_dir_path, mobile_name)
    if not os.path.exists(mobile_folder):
        os.mkdir(mobile_folder)
    # Save the top scores and correspondence to a CSV file
    output_csv_path = os.path.join(mobile_folder,f'{mobile_name}_top_scores.csv')
    with open(output_csv_path, 'w', newline='') as csvfile:
        fieldnames = ['class','uniprot','tm_score','rmsd']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for class_name, score_data in top_scores.items():            
            writer.writerow({
                'class': class_name,
                'uniprot': score_data['uniprot'],
                'tm_score': score_data['tm_score'],
                'rmsd': score_data['rmsd']
            })
            guide_name = score_data['uniprot']
            tm_score = f"{score_data['tm_score']:.3f}"
            rmsd = f"{score_data['rmsd']:.3f}"
            aligned_folder = os.path.join(mobile_folder,f"{class_name}_tm_score_{tm_score}_rmsd_{rmsd}")
            if not os.path.exists(aligned_folder):
                os.mkdir(aligned_folder)
            res = score_data['res']
            aligned_mobile_struct = transform_structure(mobile_struct, res)
            io = PDBIO()
            io.set_structure(aligned_mobile_struct)
 
            io.save(os.path.join(aligned_folder, f'{mobile_name}_to_{guide_name}_{tm_score}_{rmsd}.pdb'))
            translation = res.t
            rotation = res.u
            np.save(os.path.join(aligned_folder, f'{mobile_name}_to_{guide_name}_translation.npy'), translation)
            np.save(os.path.join(aligned_folder, f'{mobile_name}_to_{guide_name}_rotation_.npy'), rotation)
            shutil.copy(os.path.join(paths.pdb_files_structural_aligners_path, f'{guide_name}.pdb'), aligned_folder)
    return top_scores



def align_transform(mobile_pdb,guide_pdb,dir_path):
    mobile_struct = get_structure(mobile_pdb)
    mobile_coords, mobile_seq = get_residue_data(next(mobile_struct.get_chains()))
    mobile_name = os.path.splitext(os.path.basename(mobile_pdb))[0]
    guide_struct = get_structure(guide_pdb)
    guide_coords, guide_seq = get_residue_data(next(guide_struct.get_chains()))
    guide_name = os.path.splitext(os.path.basename(guide_pdb))[0]
    res = tm_align(mobile_coords, guide_coords, mobile_seq, guide_seq)
    aligned_mobile_struct = transform_structure(mobile_struct, res)
    mobile_foler = os.path.join(dir_path, mobile_name)
    if not os.path.exists(mobile_foler):
        os.mkdir(mobile_foler)
    tm_score = res.tm_norm_chain2
    rmsd = res.rmsd
    aligned_folder = os.path.join(mobile_foler,f"{guide_name}_tm_score_{tm_score:.3f}_rmsd_{rmsd:.3f}")
    if not os.path.exists(aligned_folder):
        os.mkdir(aligned_folder)
    
    io = PDBIO()
    io.set_structure(aligned_mobile_struct)
    io.save(os.path.join(aligned_folder, f'{mobile_name}_to_{guide_name}.pdb'))
    shutil.copy(guide_pdb, os.path.join(aligned_folder, f'{guide_name}.pdb'))
    translation = res.t
    rotation = res.u
    np.save(os.path.join(aligned_folder, f'{mobile_name}_to_{guide_name}_translation.npy'), translation)
    np.save(os.path.join(aligned_folder, f'{mobile_name}_to_{guide_name}_rotation_.npy'), rotation)


def convert_cif_to_pdb(input_cif_path, output_pdb_path):
    # Create parser and structure object
    parser = MMCIFParser()
    structure = parser.get_structure('structure', input_cif_path)
    
    # Create PDBIO object and save the structure to a PDB file
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_pdb_path)


if __name__ == '__main__':

    pdb_chain_table_path = os.path.join(paths.structural_aligners_path,'pdb_chain_table_uniprot_with_classes.csv')
    output_dir_path = paths.TM_aligner_aligned_pdbs_path
    
    # uniprots = ['O95793','Q8IYS0','Q86VN1','Q9NQL2','P07900','P53061']
    uniprots = ['O00487']
    for uniprot in uniprots:
        target_pdb =os.path.join(paths.AFDB_source_patch_to_score_path,'Human',f'{uniprot}.pdb') 
        top_scores = align_uniprots_with_target(target_pdb, pdb_chain_table_path,output_dir_path)
    
    # pdb_transform(target_pdb, matrix, output_path)
    # # download_alphafold_model('P02185','/home/iscb/wolfson/omriyakir/ubinet/models/structural_aligners/TM-align/aligment_examples')
    # download_alphafold_model('P69905','/home/iscb/wolfson/omriyakir/ubinet/models/structural_aligners/TM-align/aligment_examples')
    # mov_exmpale_pdb = os.path.join(paths.AFDB_source_patch_to_score_path,'Human','O00487.pdb')
    # ref_example_pdb = '/home/iscb/wolfson/omriyakir/ubinet/models/structural_aligners/TM-align/aligned_pdbs/O00487/deubiquitylase_tm_score_0.908_rmsd_1.702/P43588.pdb'
    # align_transform(mov_exmpale_pdb,ref_example_pdb,paths.TM_aligner_transformed_pdbs_path)
    