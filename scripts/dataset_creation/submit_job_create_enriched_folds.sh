#!/bin/sh
#SBATCH --job-name=enrich_folds_pts
#SBATCH --partition=cpu-killable
#SBATCH --output=./slurm_output/dataset_creation/enrich_folds/%j/logs.out
#SBATCH --time=7200 # max time (minutes)
#SBATCH --signal=USR1@120 # how to end job when time’s up
#SBATCH --nodes=1 # number of machines
#SBATCH --ntasks=1 # number of processes
#SBATCH --mem=32000 # CPU memory (MB)
#SBATCH --cpus-per-task=8 # CPU cores per process
python3 -m data_preparation.patch_to_score.v0.add_patch_coord.create_enriched_folds
