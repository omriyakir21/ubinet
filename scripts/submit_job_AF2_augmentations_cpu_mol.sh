#!/bin/sh
#SBATCH --job-name=AF2_augmentations
#SBATCH --partition=cpu-mol # (see resources section)
#SBATCH --time=14400 # max time (minutes)
#SBATCH --signal=USR1@120 # how to end job when time’s up
#SBATCH --nodes=1 # number of machines
#SBATCH --ntasks=1 # number of processes
#SBATCH --mem=1000000 # CPU memory (MB)
#SBATCH --cpus-per-task=128 # CPU cores per process
#SBATCH --output=slurm/data_preparation/ScanNet/%x_%j.out
# python3 /home/iscb/wolfson/omriyakir/ubinet/data_preparation/ScanNet/AF2_augmentations.py

colabfold_search /home/iscb/wolfson/omriyakir/ubinet/datasets/scanNet/AF2_augmentations/pdbs_with_augmentations_95/v2/fasta_folder /specific/disk2/home/mol_group/sequence_database/MMSEQS/ /home/iscb/wolfson/omriyakir/ubinet/datasets/scanNet/AF2_augmentations/pdbs_with_augmentations_95/v2/MSA_folder
