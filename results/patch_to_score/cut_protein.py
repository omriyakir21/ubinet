import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from Bio import PDB
from data_preparation.ScanNet.db_creation_scanNet_utils import THREE_LETTERS_TO_SINGLE_AA_DICT
import paths
from Bio.PDB import NeighborSearch
import argparse

class PatchSelect(PDB.Select):
    def __init__(self, patch_dict,structure, chain_id=None):
        self.patch_dict = patch_dict
        self.min_res_num, self.max_res_num = self.calculate_residue_range(structure, patch_dict, chain_id)

    def calculate_residue_range(self, structure, patch_dict, chain_id):
        chain = structure[0].child_list[0] if chain_id is None else structure[0][chain_id]
        atoms = [atom for atom in chain.get_atoms()]
        ns = NeighborSearch(atoms)
        patch_atoms = [atom for residue in chain.get_residues() if residue.get_id()[1] in patch_dict for atom in residue.get_atoms()]
        patch_indices =sorted( [residue.get_id()[1] for residue in chain.get_residues() if residue.get_id()[1] in patch_dict])
        min_patch_index = patch_indices[0]
        max_patch_index = patch_indices[-1]
        patch_len = max_patch_index - min_patch_index + 1
        if patch_len >= 150:
            return min_patch_index, max_patch_index
        
        nearby_residues = set()
        for atom in patch_atoms:
            nearby_residues.update(ns.search(atom.coord, 8.0, level='R'))
        residue_indices = sorted(residue.id[1] for residue in nearby_residues)
        if not residue_indices:
            return None, None
        if residue_indices[-1] - residue_indices[0] <= 150:
            return residue_indices[0] , residue_indices[-1]
        
        expanded_right = residue_indices[-1] - max_patch_index
        expanded_left = min_patch_index - residue_indices[0]
        left_to_expand = 150 - patch_len
        total_expanded = expanded_right + expanded_left
        max_after_reduction = max_patch_index + (left_to_expand*expanded_right)//total_expanded
        min_after_reduction = min_patch_index - (left_to_expand*expanded_left)//total_expanded
 
        return min_after_reduction, max_after_reduction

    def accept_residue(self, residue):
        res_num = residue.get_id()[1]
        if self.min_res_num is None or self.max_res_num is None:
            return False
        if self.min_res_num <= res_num <= self.max_res_num:
            return True
        return False


def extract_patch_from_pdb(id, patch, patch_number, input_file, output_dir, chain_id=None):
    patch_list = patch.split(',')
    patch_dict = {int(item[1:]): item[0] for item in patch_list}
    file_name = f"{id}_patch{str(patch_number)}.pdb"
    output_file = os.path.join(output_dir, file_name)
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure(id, input_file)

    io = PDB.PDBIO()
    io.set_structure(structure)
    output_file = os.path.join(output_dir, f"{id}_patch{str(patch_number)}.pdb")
    # Count the number of residues in the substructure
    substructure = PatchSelect(patch_dict, structure, chain_id)
    io.save(output_file, substructure) 
    residue_count = substructure.max_res_num - substructure.min_res_num + 1
    return residue_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='create continous substructure of protein using patch')
    parser.add_argument('id', type=str, help='Protein ID')
    parser.add_argument('patch', type=str, help='Patch residues')
    parser.add_argument('patch_number', type=int, help='Patch number')
    parser.add_argument('input_file', type=str, help='Input PDB file')
    parser.add_argument('output_dir', type=str, help='Output directory')

    args = parser.parse_args()

    id = args.id
    patch = args.patch
    patch_number = args.patch_number
    input_file = args.input_file
    output_dir = args.output_dir

    residue_count = extract_patch_from_pdb(id, patch, patch_number, input_file, output_dir)
    