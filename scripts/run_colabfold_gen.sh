#!/bin/bash

# Loop from 0 to 4
name="v2"  # Set the name variable to "v2"
for i in {0..4}
do
  # Submit the job with the current value of i
  sbatch scripts/submit_job_AF2_augmentations_colabfold_gpu.sh $i $name
done
