#!/bin/sh
#SBATCH --job-name=colabfold_gen
#SBATCH --partition=killable # (see resources section)
#SBATCH --time=1440 # max time (minutes)
#SBATCH --signal=USR1@120 # how to end job when time’s up
#SBATCH --nodes=1 # number of machines
#SBATCH --ntasks=1 # number of processes
#SBATCH --mem=32000 # CPU memory (MB)
#SBATCH --gpus=1 # GPU cores per process
#SBATCH --cpus-per-task=8 # CPU cores per process
#SBATCH --output=slurm/data_preparation/ScanNet/%x_%j.out
 
i=$1
name=$2

# Run colabfold_batch with the updated paths
colabfold_batch /home/iscb/wolfson/omriyakir/ubinet/datasets/scanNet/AF2_augmentations/pdbs_with_augmentations_95/${name}/MSA_parts_folder/msa_folder_${i} /home/iscb/wolfson/omriyakir/ubinet/datasets/scanNet/AF2_augmentations/pdbs_with_augmentations_95/${name}/AF2_predictions_parts/AF2_predictions_${i}
