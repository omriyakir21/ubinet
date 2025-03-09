import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from data_preparation.ScanNet.db_creation_scanNet_utils import is_ubiquitin
from Bio.PDB import PDBParser
import glob
import paths
from results.patch_to_score.patch_to_score_result_analysis import get_best_architecture_models_path,create_best_architecture_results_dir

def color_ubiquitin_chains(f, structure_path):
    parser = PDBParser()
    structure = parser.get_structure("structure", structure_path)
    for chain in structure.get_chains():
        if is_ubiquitin(chain):
            f.write(f'run(session, "color #1/{chain.id} cyan")\n')
            # f.write(f'run(session, "color sel cyan")\n')

def transperant_non_ubiquitin_chains(f, structure_path):
    parser = PDBParser()
    structure = parser.get_structure("structure", structure_path)
    for chain in structure.get_chains():
        if not is_ubiquitin(chain):
            f.write(f'run(session, "transparency #1/{chain.id} 75 target c")\n')


def create_chimera_script(folder_path):
    mobile_name = folder_path.split('/')[-2].split('_')[0]
    output_name = os.path.join(folder_path, f'{mobile_name}_chimera_script.py')
    guide_path = glob.glob(os.path.join(folder_path, "*ubiqs.pdb"))[0]
    with open(output_name, 'w') as f:
        f.write('import os\n')  
        f.write('from chimerax.core.commands import run\n')
        f.write('import glob\n')
        f.write('folder_path = os.path.dirname(os.path.abspath(__file__))\n')
        f.write('guide_path = glob.glob(os.path.join(folder_path, "*ubiqs.pdb"))[0]\n')
        f.write('mobile_path = glob.glob(os.path.join(folder_path, "*to_*.pdb"))[0]\n')
        f.write('mobile_name = "{mobile_name}"\n'.format(mobile_name=mobile_name))
        f.write('run(session,"set bgColor #ffffff00")\n')
        f.write('run(session,"open {guide_path}".format(guide_path=guide_path))\n')
        f.write('run(session,"open {mobile_path}".format(mobile_path=mobile_path))\n')
        f.write('run(session,"color byattribute bfactor #2 palette blue:white:red range 0,70")\n')
        f.write('run(session,"color #1 forest green")\n')
        color_ubiquitin_chains(f, guide_path)
        transperant_non_ubiquitin_chains(f, guide_path)
        f.write('run(session,"alphafold fetch {mobile_name}".format(mobile_name=mobile_name))\n')
        f.write('run(session,"matchmaker #3 to #2")\n')
        f.write('run(session,"hide #3 models")\n')
    return output_name

def process_folders(root_folder):
    subfolders = [f.path for f in os.scandir(root_folder) if f.is_dir()]
    for subfolder in subfolders:
        subsubfolders = [f.path for f in os.scandir(subfolder) if f.is_dir()]
        for subsubfolder in subsubfolders:
            for file in os.scandir(subsubfolder):
                if file.is_file() and file.name.endswith('chimera_script.py'):
                    chimera_script_path = file.path
                    break
            os.remove(chimera_script_path)
            print(f"Removed {chimera_script_path}")
            create_chimera_script(subsubfolder)


if __name__ == "__main__":
    best_architecture_models_path = get_best_architecture_models_path(paths.with_MSA_50_plddt_0304_models_dir, paths.with_MSA_50_plddt_0304_results_dir)
    best_architecture_results_dir = create_best_architecture_results_dir(best_architecture_models_path,paths.with_MSA_50_plddt_0304_results_dir)
    process_folders(os.path.join(best_architecture_results_dir,'aligned_chainsaw_pdbs_with_ubiqs','patch_1'))