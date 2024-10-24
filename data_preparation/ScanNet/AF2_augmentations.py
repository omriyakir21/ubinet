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
import numpy as np
import logging
from create_tables_and_weights import cluster_sequences
import biotite.sequence as seq
import biotite.sequence.io.fasta as fasta
from biotite.application.blast import BlastWebApp
import logging
import os
import numpy as np
from create_tables_and_weights import cluster_sequences
from Bio.PDB import MMCIFParser, MMCIFIO, Select
import pandas as pd

class ChainSelect(Select):
    def __init__(self, chain_id):
        self.chain_id = chain_id

    def accept_chain(self, chain):
        return chain.id == self.chain_id

def process_pdb_chain_table(csv_path, structures_path, new_folder_path):
    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Create the new folder path if it doesn't exist
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)

    # Initialize the MMCIF parser
    parser = MMCIFParser(QUIET=True)

    for index, row in df.iterrows():
        pdb_id = row['PDB_ID']
        chain_id = row['CHAIN_ID']
        
        # Create a new folder for each PDB_ID_CHAIN_ID
        new_subfolder = os.path.join(new_folder_path, f"{pdb_id}_{chain_id}")
        if not os.path.exists(new_subfolder):
            os.makedirs(new_subfolder)
        
        # Parse the structure file
        structure_file = os.path.join(structures_path, f"{pdb_id}.cif")
        structure = parser.get_structure(pdb_id, structure_file)
        
        # Select the chain and save the substructure as CIF
        io = MMCIFIO()
        io.set_structure(structure)
        io.save(os.path.join(new_subfolder, f"{pdb_id}_{chain_id}.cif"), ChainSelect(chain_id))


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

def run_blast_online(query_sequence, db, output, evalue=1e-5,hitlist_size=50,entrez_query="(none)",matrix_name="BLOSUM62"):
    result_handle = NCBIWWW.qblast("blastp", db, query_sequence, expect=evalue, format_type="XML",
                                   hitlist_size=hitlist_size,entrez_query=entrez_query,matrix_name=matrix_name)
    with open(output, "w") as out_handle:
        out_handle.write(result_handle.read())

def run_blast_biotite(query_sequence, db, output, evalue=1e-5, hitlist_size=50, entrez_query="(none)", matrix_name="BLOSUM62"):
    app = BlastWebApp(
        program="blastp",
        query=query_sequence,
        database=db,
        mail="padix.key@gmail.com"
    )
    app.set_max_expect_value(evalue)
    app.set_max_results(hitlist_size)
    app.set_entrez_query(entrez_query)
    app.set_substitution_matrix(matrix_name)
    
    app.start()
    app.join()
    
    with open(output, "w") as out_handle:
        out_handle.write(app.get_xml_response())

def find_close_homologs_biotite(id, sequence, output_folder):
    logging.info(f"Finding close homologs for {id}, with sequence {sequence}")
    output_path = os.path.join(output_folder, f"{id}_blast_results.xml")
    
    if not os.path.exists(output_path):
        attempts = 0
        max_attempts = 2
        while attempts < max_attempts:
            try:
                run_blast_biotite(sequence, "nr", output_path)
                break
            except Exception as e:
                attempts += 1
                if attempts == max_attempts:
                    logging.error(f"Failed to run BLAST with Biotite after {max_attempts} attempts for id {id}, error message: {e}")
                    return None
    
    # Parse BLAST results
    app = BlastWebApp("blastp", sequence)
    app.start()
    app.join()
    alignments = app.get_alignments()

    all_sequences = []
    for alignment in alignments:
        similarity = alignment.identities / alignment.length
        if similarity < 0.95 or alignment.hit_sequence.replace("-", "") == sequence:
            continue
        all_sequences.append({
            "aligned_sequence": alignment.hit_sequence.replace("-", ""),
            "definition": alignment.hit_def,
            "similarity": similarity
        })
    
    if not all_sequences:
        return None
    
    logging.info(f"Found {len(all_sequences)} homologs for {id}")
    
    # Cluster the sequences and take cluster representatives
    cluster_indices, representative_indices = cluster_sequences([all_sequences[i]["aligned_sequence"] for i in range(len(all_sequences))],
                                                                 seqid=0.98, coverage=0.9, covmode='0')
    
    cluster_sizes = {i: np.sum(cluster_indices == i) for i in set(cluster_indices)}
    
    sorted_representative_indices = sorted(representative_indices, key=lambda x: cluster_sizes[list(representative_indices).index(x)], reverse=True)
    top_representatives = sorted_representative_indices[:min(10, len(sorted_representative_indices))]
    
    homologs = [all_sequences[i] for i in top_representatives]
    
    # Save the cluster representatives as the final paralogs
    output_file = os.path.join(output_folder, f"{id}_homologs.csv")
    with open(output_file, "w") as csvfile:
        csvfile.write("sequence,similarity,definition\n")
        for i in range(len(homologs)):
            result = homologs[i]
            csvfile.write(f"{result['aligned_sequence']},{result['similarity']:.2f},{result['definition']}\n")
    
    return homologs


def find_all_close_homologs(input_folder):
    cnt = 0
    for subfolder in os.listdir(input_folder):
        subfolder_path = os.path.join(input_folder, subfolder)
        if os.path.isdir(subfolder_path):
            pdb_file = os.path.join(subfolder_path, f"{subfolder}.cif")
            if os.path.exists(pdb_file):
                # Extract the sequence from the PDB file
                for record in SeqIO.parse(pdb_file, "cif-atom"):
                    sequence = str(record.seq)
                    break
                # Call find_close_homologs with the id, sequence, and subfolder path as output folder
                # find_close_homologs_biotite(subfolder, sequence, subfolder_path)
                logging.info(f"Processing {subfolder}")
                find_close_homologs(subfolder, sequence, subfolder_path)
                logging.info(f"Processed {subfolder}, cnt: {cnt}")
                cnt += 1

def create_fasta_file(sequence, id,output_folder):
    fasta_file = os.path.join(output_folder, f"{id}.fasta")
    with open(fasta_file, "w") as out_handle:
        out_handle.write(f">{id}\n{sequence}\n")
    return fasta_file


def find_close_homologs(id,sequence, output_folder):
    logging.info(f"Finding close homologs for {id}, with sequence {sequence}")
    # Run BLAST command with entrez_query to filter by organism
    output_path = os.path.join(output_folder, f"{id}_blast_results.xml") 
    if not os.path.exists(output_path):
        attempts = 0
        max_attempts = 2
        while attempts < max_attempts:
            try:
                run_blast_online(sequence, "nr", output_path)
                break
            except Exception as e:
                attempts += 1
                if attempts == max_attempts:
                    logging.error(f"Failed to run BLAST online after {max_attempts} attempts for id {id}, error message: {e}")
                    return None

    output_file = os.path.join(output_folder, f"{id}_homologs.csv")
    if os.path.exists(output_file):
        return None
    # Parse BLAST results
    with open(output_path) as result_handle:
        blast_records = NCBIXML.parse(result_handle)
        
        all_sequences = []
        for blast_record in blast_records:
            for alignment in blast_record.alignments:
                for hsp in alignment.hsps:
                    similarity = hsp.identities / hsp.align_length
                    if similarity < 0.95 or hsp.sbjct.replace("-", "") == sequence: 
                        continue
                    all_sequences.append({
                        "aligned_sequence": hsp.sbjct.replace("-", ""),
                        "definition": alignment.hit_def,
                        "similarity": similarity
                    })
            
        if len(all_sequences) == 0:
            with open(output_file, "w") as csvfile:
                csvfile.write("sequence,similarity,definition\n")
            logging.info(f"No homologs found for {id}")
            return None

        logging.info(f"Found {len(all_sequences)} homologs for {id}")
        # Cluster the sequences and take cluster representatives
        cluster_indices, representative_indices = cluster_sequences([all_sequences[i]["aligned_sequence"] for i in range(len(all_sequences))],
                                                        seqid=0.98,coverage=0.9, covmode='0')
        print(f'cluster_indices {cluster_indices}')
        print(f'representative_indices {representative_indices}')
        # Count the size of each cluster
        cluster_sizes = {i: np.sum(cluster_indices == i) for i in set(cluster_indices)}
        print(f'cluster_sizes {cluster_sizes}')

        # Sort the representative indices by cluster size in descending order
        sorted_representative_indices = sorted(representative_indices, key=lambda x:cluster_sizes[list(representative_indices).index(x)], reverse=True)

        # Take the top ten representatives
        top_representatives = sorted_representative_indices[:min(10, len(sorted_representative_indices))]
        
        homologs = [all_sequences[i] for i in top_representatives]

        # Save the cluster representatives as the final paralogs
        with open(output_file, "w") as csvfile:
            csvfile.write("sequence,similarity,definition\n")
            for i in range(len(homologs)):
                result = homologs[i]
                csvfile.write(f"{result['aligned_sequence']},{result['similarity']:.3f},{result['definition']}\n")
                create_fasta_file(result['aligned_sequence'], f'{id}_{i}', output_folder)
    return homologs

def clean_subfolders(folder_path):
    # Iterate through all subfolders in the given folder
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # Get the file extension
            file_extension = os.path.splitext(file)[1]
            # Check if the file extension is not .pdb or .xml
            if file_extension not in ['.pdb', '.xml']:
                # Construct the full file path
                file_path = os.path.join(root, file)
                # Remove the file
                os.remove(file_path)
                print(f"Removed: {file_path}")

if __name__ == '__main__':
    DATE = '8_9'
    # pssm_content_path = os.path.join(paths.PSSM_path,f'PSSM_{DATE}', 'propagatedPssmWithAsaFile.txt')
    # asa_content_path = os.path.join(paths.ASA_path, 'normalizedFullASAPssmContent.txt')
    # _, _, _, chain_names, _, _ = split_receptors_into_individual_chains(
    #     pssm_content_path, asa_content_path)
    # chain_dict = make_chain_dict(chain_names)
    # from_chains_dict_to_pdb_chain_table(chain_dict)
    # process_pdb_chain_table(os.path.join(paths.scanNet_AF2_augmentations_path,'pdb_chain_table.csv'), paths.chosen_assemblies_path, paths.pdbs_with_augmentations_path)
    log_file = os.path.join(paths.scanNet_AF2_augmentations_path, f"AF2_augmentations_2.log")
    logging.basicConfig(filename=log_file, level=logging.INFO, 
                        format='%(asctime)s %(levelname)s:%(message)s')
    find_all_close_homologs(paths.pdbs_with_augmentations_path) 
    # clean_subfolders(paths.original_pdbs_with_augmentations_path)