#!/bin/sh
#SBATCH --job-name=cath
#SBATCH --partition=cpu-mol # (see resources section)
#SBATCH --time=14400 # max time (minutes)
#SBATCH --signal=USR1@120 # how to end job when time’s up
#SBATCH --nodes=1 # number of machines
#SBATCH --ntasks=1 # number of processes
#SBATCH --mem=1000000 # CPU memory (MB)
#SBATCH --cpus-per-task=128 # CPU cores per process
#SBATCH --output=slurm/data_preparation/ScanNet/%x_%j.out
python3 /home/iscb/wolfson/omriyakir/ubinet/data_preparation/ScanNet/cath.py

