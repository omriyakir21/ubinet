#!/bin/sh
#SBATCH --job-name=queen
#SBATCH --partition=killable # (see resources section)
#SBATCH --time=1440 # max time (minutes)
#SBATCH --signal=USR1@120 # how to end job when time’s up
#SBATCH --nodes=1 # number of machines
#SBATCH --ntasks=1 # number of processes
#SBATCH --mem=32000 # CPU memory (MB)
#SBATCH --gpus=1 # GPU cores per process
#SBATCH --cpus-per-task=8 # CPU cores per process
#SBATCH --output=slurm/models/QUEEN/%x_%j.out   
python3 /home/iscb/wolfson/omriyakir/ubinet/models/QUEEN/queen.py
