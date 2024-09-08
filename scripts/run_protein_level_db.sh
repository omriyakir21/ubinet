#!/bin/bash

CLASS_NAME=DUB
BATCH_SIZE=2000

# Load the uniprot names dict
UNIPROT_NAMES_DICT=$(python3 -c "import pickle; import os; import paths; print(pickle.load(open(os.path.join(paths.GO_source_patch_to_score_path, 'uniprotNamesDictNew.pkl'), 'rb'))['$CLASS_NAME'])")
TOTAL_UNIPROTS=$(echo $UNIPROT_NAMES_DICT | wc -w)

#submit jobs in batches

for ((i=o; i<$TOTAL_UNIPROTS; i+=$BATCH_SIZE)); do
	j=$((i + $BATCH_SIZE))
	if [ $j -gt $TOTAL_UNIPROTS ]; then
		j=$TOTAL_UNIPROTS
	fi
	sbatch scripts/submit_job_cpu_protein_level_db.sh $CLASS_NAME $i $j
done
