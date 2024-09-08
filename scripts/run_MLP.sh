#!/bin/bash


# Loop through the specified ranges for n_layers, m_a, and m_c
# for n_layers in 4 5; do
#   for m_a in 128 256 512; do
    # for m_c in 128 256 512; do
for n_layers in 4 5; do
  for m_a in 512 1024; do
    for m_c in 512 1024; do
      # Call sbatch with the current parameters
      sbatch scripts/submit_job_MLP.sh $n_layers $m_a $m_c
    done
  done
done