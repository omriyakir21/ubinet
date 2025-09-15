#!/bin/sh
#SBATCH --job-name=af2_augmentations
#SBATCH --partition=gpu-mol # (see resources section)
#SBATCH --time=1440 # max time (minutes)
#SBATCH --signal=USR1@120 # how to end job when time’s up
#SBATCH --nodes=1 # number of machines
#SBATCH --ntasks=1 # number of processes
#SBATCH --mem=64000 # CPU memory (MB)
#SBATCH --cpus-per-task=16 # CPU cores per process
#SBATCH --gpus=1 # GPU cores per process
#SBATCH --output=slurm/data_preparation/ScanNet/%x_%j.out
python3 /home/iscb/wolfson/omriyakir/ubinet/data_preparation/ScanNet/AF2_augmentations.py
# colabfold_batch /home/iscb/wolfson/omriyakir/ubinet/datasets/scanNet/AF2_augmentations/pdbs_with_augmentations_95/MSA_folder /home/iscb/wolfson/omriyakir/ubinet/datasets/scanNet/AF2_augmentations/pdbs_with_augmentations_95/AF2_predictions 