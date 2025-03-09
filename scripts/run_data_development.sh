#!/bin/bash

for i in {0..0}
do
  sbatch scripts/submit_job_cpu_data_development.sh $i
done
