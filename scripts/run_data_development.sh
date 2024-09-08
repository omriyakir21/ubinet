#!/bin/bash

for i in {46..87}
do
  sbatch scripts/submit_job_cpu_data_development.sh $i
done
