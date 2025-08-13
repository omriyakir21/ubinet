#!/bin/sh
#SBATCH --job-name=db_creation_scanNet
#SBATCH --partition=cpu-killable # (see resources section)
#SBATCH --time=7200 # max time (minutes)
#SBATCH --signal=USR1@120 # how to end job when time’s up
#SBATCH --nodes=1 # number of machines
#SBATCH --ntasks=1 # number of processes
#SBATCH --mem=32000 # CPU memory (MB)
#SBATCH --cpus-per-task=8 # CPU cores per process
#SBATCH --output=slurm/data_preparation/ScanNet/%x_%j.out
python3 /home/iscb/wolfson/omriyakir/ubinet/data_preparation/ScanNet/db_creation_scanNet.py $1
