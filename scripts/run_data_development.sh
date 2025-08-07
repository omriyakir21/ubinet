#!/bin/bash

for i in {0..100}
do
  sbatch scripts/submit_job_cpu_data_development.sh $i
done
