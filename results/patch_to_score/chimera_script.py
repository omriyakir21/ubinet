import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from data_preparation.ScanNet.db_creation_scanNet_utils import is_ubiquitin
from Bio.PDB import PDBParser
import glob

def color_ubiquitin_chains(f, structure_path):
    parser = PDBParser()
    structure = parser.get_structure("structure", structure_path)
    for chain in structure.get_chains():
        if is_ubiquitin(chain):
            f.write(f'run(session, "sel #1/ {chain.id}")\n')
            f.write(f'run(session, "color sel forest green")\n')



def create_chimera_script(folder_path):
    output_name = os.path.join(folder_path, 'chimera_script.py')
    guide_path = glob.glob(os.path.join(folder_path, "*ubiqs.pdb"))[0]
    with open(output_name, 'w') as f:
        f.write('import os\n')  
        f.write('from chimerax.core.commands import run\n')
        f.write('import glob\n')

        f.write('folder_path = os.path.dirname(os.path.abspath(__file__))\n')
        f.write('guide_path = glob.glob(os.path.join(folder_path, "*ubiqs.pdb"))[0]\n')
        f.write('mobile_path = glob.glob(os.path.join(folder_path, "*to_*.pdb"))[0]\n')
        f.write('run(session,"set bgColor #ffffff00")\n')
        f.write('run(session,"open {guide_path}".format(guide_path=guide_path))\n')
        f.write('run(session,"open {mobile_path}".format(mobile_path=mobile_path))\n')
        f.write('run(session,"sel #1")\n')
        f.write('run(session,"color sel blue ")\n')
        f.write('run(session,"sel #2")\n')
        f.write('run(session,"color sel red ")\n')
        color_ubiquitin_chains(f, guide_path)
    return output_name

# if __name__ == "__main__":
#     folder_path = '/home/iscb/wolfson/omriyakir/ubinet/results/patch_to_score/all_predictions_0304/with_MSA_50_plddt/architecture:5_1024_1024_1024/aligned_pdbs_with_ubiqs/A0A0A6YYL3_patch1/deubiquitylase_tm_score_0.756_rmsd_0.948/'
#     create_chimera_script(folder_path)