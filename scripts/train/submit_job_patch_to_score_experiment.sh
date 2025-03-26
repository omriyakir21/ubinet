#!/bin/sh
#SBATCH --job-name=train_patch_to_score
#SBATCH --partition=gpu-h100-killable # (see resources section)
#SBATCH --output=./slurm_output/train_patch_to_score/%j/logs.out
#SBATCH --time=1440 # max time (minutes)
#SBATCH --signal=USR1@120 # how to end job when time’s up
#SBATCH --nodes=1 # number of machines
#SBATCH --ntasks=1 # number of processes
#SBATCH --mem=32000 # CPU memory (MB)
#SBATCH --gpus=1 # GPU cores per process
#SBATCH --cpus-per-task=8 # CPU cores per process
python3 models/patch_to_score/train.py --config $1
