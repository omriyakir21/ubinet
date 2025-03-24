#!/bin/sh
#SBATCH --job-name=copy_results
#SBATCH --partition=cpu-killable # (see resources section)
#SBATCH --output=./slurm_output/copy_results/%j/logs.out
#SBATCH --time=1440 # max time (minutes)
#SBATCH --signal=USR1@120 # how to end job when time’s up
#SBATCH --nodes=1 # number of machines
#SBATCH --ntasks=1 # number of processes
#SBATCH --mem=32000 # CPU memory (MB)
#SBATCH --cpus-per-task=8 # CPU cores per process
cp -R /home/iscb/wolfson/omriyakir/ubinet/results/ /home/iscb/wolfson/doririmon/home/order/ubinet/repo/ubinet
