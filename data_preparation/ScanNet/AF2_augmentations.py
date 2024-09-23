import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import paths
import csv
from results.ScanNet.orientation_based_performence_analysis import split_receptors_into_individual_chains,make_chain_dict
import shutil
from Bio.Blast import NCBIXML
from Bio.Blast import NCBIWWW
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
import numpy as np
from sklearn.cluster import AgglomerativeClustering

def from_chains_dict_to_pdb_chain_table(chain_dict):
    with open(os.path.join(paths.scanNet_AF2_augmentations_path,'pdb_chain_table.csv'), mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header row
        writer.writerow(['PDB_ID', 'CHAIN_ID'])
        # Iterate over the chain_dict
        for pdb_chain, chains in chain_dict.items():
            pdb_id = pdb_chain.split('_')[0]
            for chain in chains:
                writer.writerow([pdb_id, chain])

def create_original_pdbs_folders(input_folder, output_folder):
    # Iterate over all files in the input folder
    for filename in os.listdir(input_folder):
        # Check if the file is a .pdb file
        if filename.endswith('.pdb'):
            # Extract the ID from the filename (remove the .pdb extension)
            pdb_id = filename.rsplit('.', 1)[0]
            # Create the new folder path
            new_folder_path = os.path.join(output_folder, pdb_id)
            # Create the new folder
            os.makedirs(new_folder_path, exist_ok=True)
            # Copy the .pdb file to the new folder
            shutil.copy(os.path.join(input_folder, filename), new_folder_path)

def run_blast_online(query_sequence, db, output, evalue=1e-5,hitlist_size=50,entrez_query="(none)",matrix_name="BLOSUM62"):
    result_handle = NCBIWWW.qblast("blastp", db, query_sequence, expect=evalue, format_type="XML",
                                   hitlist_size=hitlist_size,entrez_query=entrez_query,matrix_name=matrix_name)
    with open(output, "w") as out_handle:
        out_handle.write(result_handle.read())

def find_all_close_homologs(input_folder):
    for subfolder in os.listdir(input_folder):
        subfolder_path = os.path.join(input_folder, subfolder)
        if os.path.isdir(subfolder_path):
            pdb_file = os.path.join(subfolder_path, f"{subfolder}.pdb")
            if os.path.exists(pdb_file):
                # Extract the sequence from the PDB file
                for record in SeqIO.parse(pdb_file, "pdb-atom"):
                    sequence = str(record.seq)
                    break
                # Call find_close_homologs with the id, sequence, and subfolder path as output folder
                find_close_homologs(subfolder, sequence, subfolder_path)

def find_close_homologs(id,sequence, output_folder):
    # Run BLAST command with entrez_query to filter by organism
    output_path = os.path.join(output_folder, f"{id}_blast_results.xml")
    if not os.path.exists(output_path):
        attempts = 0
        max_attempts = 5
        while attempts < max_attempts:
            try:
                run_blast_online(sequence, "nr", output_path)
                break
            except Exception as e:
                attempts += 1
                if attempts == max_attempts:
                    raise Exception(f"Failed to run BLAST online after {max_attempts} attempts") from e
                print(f"Attempt {attempts} failed: {e}. Retrying...")

    # Parse BLAST results
    with open(output_path) as result_handle:
        blast_records = NCBIXML.parse(result_handle)
        
        all_sequences = []
        results = []
        for blast_record in blast_records:
            for alignment in blast_record.alignments:
                for hsp in alignment.hsps:
                    similarity = hsp.identities / hsp.align_length
                    if similarity < 0.95:
                        continue
                    all_sequences.append({
                        "aligned_sequence": hsp.sbjct,
                        "definition": alignment.hit_def,
                        "similarity": similarity
                    })

        if not all_sequences:
            return None

        # Cluster the sequences and take cluster representatives
        cluster_indices, representative_indices = cluster_sequences([all_sequences[i]["aligned_sequence"].replace("-", "") for i in range(len(all_sequences))],
                                                        seqid=0.98,coverage=0.9, covmode='0')
        # Count the size of each cluster
        cluster_sizes = {i: cluster_indices.count(i) for i in set(cluster_indices)}

        # Sort the representative indices by cluster size in descending order
        sorted_representative_indices = sorted(representative_indices, key=lambda x: cluster_sizes[x], reverse=True)

        # Take the top ten representatives
        top_representatives = sorted_representative_indices[:min(10, len(sorted_representative_indices))]
        
        homologs = [all_sequences[i] for i in top_representatives]

        # Save the cluster representatives as the final paralogs
        output_file = os.path.join(output_folder, f"{organism}_paralogs.csv")
        with open(output_file, "w") as csvfile:
            csvfile.write("sequence,similarity,definition\n")
            for result in homologs:
                csvfile.write(f"{result['aligned_sequence']},{result['similarity']:.2f},{result['definition']}\n")

    return paralogs
if __name__ == '__main__':
    # pssm_content_path = os.path.join(paths.PSSM_path, 'propagatedPssmWithAsaFile.txt')
    # asa_content_path = os.path.join(paths.ASA_path, 'normalizedFullASAPssmContent.txt')
    # _, _, _, chain_names, _, _ = split_receptors_into_individual_chains(
    #     pssm_content_path, asa_content_path)
    # chain_dict = make_chain_dict(chain_names)
    # from_chains_dict_to_pdb_chain_table(chain_dict)
    # create_original_pdbs_folders(paths.pdb_files_structural_aligners_path,paths.scanNet_AF2_augmentations_path)