#!/bin/bash

# Loop from 1 to 39
for i in {21..39}
do
  # Submit the job with the current value of i
  sbatch scripts/submit_job_cpu_db_creation.sh $i
done
