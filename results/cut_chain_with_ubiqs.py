import os
import sys 
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data_preparation.ScanNet.db_creation_scanNet_utils import is_ubiquitin
from Bio import PDB
import paths
import pandas as pd

def create_substructures_for_all_chains(input_folder,chains_info_csv):
    chains_info = pd.read_csv(chains_info_csv)
    for _, row in chains_info.iterrows():
        pdb_id = row['PDB_ID']
        chain_id = row['CHAIN_ID']
        print(f"Processing {pdb_id} ,{chain_id}")
        if len(chain_id) > 1:
            print(f"Skipping {pdb_id} {chain_id}")
            continue
        input_path = os.path.join(input_folder, f"{pdb_id}.cif")
        
        output_path_with_ubiqs = os.path.join(paths.binding_chains_pdbs_with_ubiqs_path, f"{pdb_id}_{chain_id}_with_ubiqs.pdb")
        extract_chains(input_path, chain_id, output_path_with_ubiqs, include_ubiquitin=True, input_type='cif')
        output_path_without_ubiqs = os.path.join(paths.binding_chains_pdbs_without_ubiqs_path, f"{pdb_id}_{chain_id}.pdb")
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

if __name__ == "__main__":
    structures_folder = paths.chosen_assemblies_path
    chains_info_csv = os.path.join(paths.structural_aligners_path, 'pdb_chain_table_uniprot_with_classes.csv')
    create_substructures_for_all_chains(structures_folder, chains_info_csv)