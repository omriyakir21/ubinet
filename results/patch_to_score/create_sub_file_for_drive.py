import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import paths
import shutil
import pandas as pd


# Function to copy relevant files
def copy_relevant_files(src, dst, uniprot_id):
    for root, dirs, files in os.walk(src):
        for file in files:
            if file.endswith('.pdb') or file.endswith('.py'):
                src_file = os.path.join(root, file)
                dst_file = os.path.join(dst, os.path.relpath(src_file, src))
                os.makedirs(os.path.dirname(dst_file), exist_ok=True)
                shutil.copy2(src_file, dst_file)




if __name__ == "__main__":
    csv_file = '/home/iscb/wolfson/omriyakir/ubinet/results/patch_to_score/all_predictions_0304/with_MSA_50_plddt/architecture:5_1024_1024_1024/data_predictions_significances_final_Human_with_tm_info.csv'
    main_folder = '/home/iscb/wolfson/omriyakir/ubinet/results/patch_to_score/all_predictions_0304/with_MSA_50_plddt/architecture:5_1024_1024_1024/aligned_chainsaw_pdbs_with_ubiqs'
    new_main_folder = '/home/iscb/wolfson/omriyakir/ubinet/results/patch_to_score/all_predictions_0304/with_MSA_50_plddt/architecture:5_1024_1024_1024/data_for_gali'
    # Read the top 50 Uniprot IDs from the CSV file
    df = pd.read_csv(csv_file)
    top_50_uniprot_ids = df['uniprot'].head(50).tolist()

    if not os.path.exists(new_main_folder):
        os.makedirs(new_main_folder)
        # Iterate over patch1 and patch2 folders
    for patch in ['1', '2']:
        patch_folder = os.path.join(main_folder, f'patch_{patch}')
        for uniprot_id in top_50_uniprot_ids:
            src_folder = os.path.join(patch_folder, f'{uniprot_id}_patch{patch}')
            print(f'Checking {src_folder}')
            dst_folder = os.path.join(new_main_folder, patch, f'{uniprot_id}_patch{patch}')
            if os.path.exists(src_folder):
                print(f'Copying files from {src_folder} to {dst_folder}')
                copy_relevant_files(src_folder, dst_folder, uniprot_id)