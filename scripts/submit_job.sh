#!/bin/sh
#SBATCH --job-name=gpu
#SBATCH --partition=killable # (see resources section)
#SBATCH --time=1440 # max time (minutes)
#SBATCH --signal=USR1@120 # how to end job when time’s up
#SBATCH --nodes=1 # number of machines
#SBATCH --ntasks=1 # number of processes
#SBATCH --mem=32000 # CPU memory (MB)
#SBATCH --gpus=1 # GPU cores per process
#SBATCH --cpus-per-task=8 # CPU cores per process
#SBATCH --constraint="tesla_v100|quadro_rtx_8000|geforce_rtx_3090|a100|a5000|a6000"
python3 
