#!/bin/bash

# Loop from 1 to 39
for i in {51..100}
do
  # Submit the job with the current value of i
  sbatch scripts/submit_job_cpu_label_propagation.sh $i
done
