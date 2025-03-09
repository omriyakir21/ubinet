import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from Bio import PDB
from data_preparation.ScanNet.db_creation_scanNet_utils import THREE_LETTERS_TO_SINGLE_AA_DICT
import paths
from Bio.PDB import NeighborSearch
import argparse
import pandas as pd

class PatchSelect(PDB.Select):
    def __init__(self, patch_dict,structure,structure_path,all_predictions_ubiq, chain_id=None):
        self.patch_dict = patch_dict
        self.min_res_num, self.max_res_num = self.calculate_residue_range(structure, patch_dict, chain_id,structure_path)
        chain = structure[0].child_list[0] if chain_id is None else structure[0][chain_id]
        def color_chain(chain,all_predictions_ubiq):
            residues = [residue for residue in chain.get_residues()]
            assert len(residues) == len(all_predictions_ubiq)
            for i in range(len(residues)):
                residue = residues[i]
                atoms = [atom for atom in residue.get_atoms()]
                for atom in atoms:
                    atom.set_bfactor(all_predictions_ubiq[i]*100)
        color_chain(chain,all_predictions_ubiq)


    
    def calculate_residue_range(self, structure, patch_dict, chain_id,structure_path):
        chain = structure[0].child_list[0] if chain_id is None else structure[0][chain_id]
        patch_indices =sorted( [residue.get_id()[1] for residue in chain.get_residues() if residue.get_id()[1] in patch_dict])
        min_patch_index = patch_indices[0]
        max_patch_index = patch_indices[-1]
        chainsaw_output_path = os.path.join(paths.chainsaw_results_path,structure_path.split('/')[-1].split('.')[0] + '_chainsaw.tsv')
        old_cwd = os.getcwd()
        if not os.path.exists(chainsaw_output_path):
            try:
                os.chdir('/home/iscb/wolfson/omriyakir/chainsaw')
                os.system(f'python get_predictions.py --structure_file {structure_path} --output {chainsaw_output_path}')
            finally:
                os.chdir(old_cwd)
        chainsaw_output = pd.read_csv(chainsaw_output_path, sep='\t')
        print(f'chainsaw_output: {chainsaw_output}')
        print(f'chainsaw_output[chopping]: {chainsaw_output["chopping"]}')
        if type(chainsaw_output['chopping'][0]) != str:
            print(f'No chopping found for structure {structure.id}')
            return min_patch_index, max_patch_index
        choppings = chainsaw_output['chopping'][0].split(',')
        print(f'choppings: {choppings}')
        
        domains = []
        for chop in choppings:
            for domain in chop.split('_'):
                start_index = int(domain.split('-')[0])
                end_index = int(domain.split('-')[1])
                domains.append((start_index,end_index))

        included_domains = []
        for domain in domains:
            if domain[0] <= min_patch_index and domain[1] >= max_patch_index:
                included_domains.append(domain)
        included_domains.append((min_patch_index,max_patch_index))
        min_index_included = min([domain[0] for domain in included_domains])
        max_index_included = max([domain[1] for domain in included_domains])
        return min_index_included, max_index_included


    def accept_residue(self, residue):
        res_num = residue.get_id()[1]
        if self.min_res_num is None or self.max_res_num is None:
            return False
        if self.min_res_num <= res_num <= self.max_res_num:
            return True
        return False


def extract_patch_from_pdb(id, patch, patch_number, input_file, output_dir,all_predictions_ubiq, chain_id=None):
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
    substructure = PatchSelect(patch_dict, structure,input_file,all_predictions_ubiq, chain_id)
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
    