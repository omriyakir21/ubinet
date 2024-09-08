#!/bin/sh

input_dir="/home/iscb/wolfson/omriyakir/ubinet/models/structural_aligners/missing_uniprots_fasta_files"
output_dir="/home/iscb/wolfson/omriyakir/ubinet/models/structural_aligners/files_from_colab_fold"
script_path="/home/iscb/wolfson/omriyakir/ubinet/scripts/submit_job_colab_fold.sh"

for input_file in "$input_dir"/*.fasta; do
  filename=$(basename -- "$input_file")
  filename_without_ext="${filename%.fasta}"
  output_path="${output_dir}/${filename_without_ext}"
  if [ ! -f "$output_path" ]; then
    sbatch "$script_path" "$input_file" "$output_path"
  fi
done