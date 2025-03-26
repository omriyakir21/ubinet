#!/bin/bash
args=("$@")
# echo $# arguments passed
hypothesis=${args[0]}
dir_path="configurations/data/"$hypothesis
experiment_amount=$(ls -1 $dir_path/*.json | wc -l)
echo running all $experiment_amount experiments under hypothesis: $hypothesis
for filename in $dir_path/*.json; do
    echo running experiment: $filename
    sbatch scripts/train/submit_job_patch_to_score_experiment $filename
done