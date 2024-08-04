import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import paths
import pickle
import subprocess
import shutil
from builtins import classmethod

import numpy as np
import pandas as pd
import requests
from Bio.PDB import MMCIFParser
import csv
from data_preparation.ScanNet.db_creation_scanNet_utils import save_as_pickle, load_as_pickle, aa_out_of_chain, \
    get_str_seq_of_chain
from data_preparation.ScanNet.LabelPropagationAlgorithm_utils import cluster_sequences
from pyparsing import unicode_set
from tensorflow.python.tpu.tpu_embedding_v2 import extract_variable_info
import requests, sys
import re

import path as path


def get_uniprot_ids_from_gpad_file(path):
    uniprot_ids = []
    with open(path, 'r') as gpad_file:
        for line in gpad_file:
            fields = line.strip().split('\t')
            if len(fields) >= 8 and fields[0] == 'UniProtKB':
                uniprot_ids.append(fields[1])
    return uniprot_ids


def get_evidence_from_gpad_file(path):
    evidence_list = []
    with open(path, 'r') as gpad_file:
        for line in gpad_file:
            fields = line.strip().split('\t')
            if len(fields) >= 8 and fields[0] == 'UniProtKB':
                evidence_list.append(fields[11].split('=')[1])
    return evidence_list


def get_uniprot_ids_util(path):
    uniprot_ids = []
    # Open and read the text file
    with open(path, 'r') as file:
        for line in file:
            # Split each line by tabs or spaces
            parts = line.strip().split('\t')  # You can also use split(' ') if space-separated
            # Check if there are at least two parts in the line
            if len(parts) >= 2:
                # Extract the UniProt ID from the first part
                uniprot_id = parts[0].strip().split(':')[1]
                # Append the UniProt ID to the list
                uniprot_ids.append(uniprot_id)

    return uniprot_ids


def get_evidence_util(path):
    evidence_list = []
    # Open and read the text file
    with open(path, 'r') as file:
        for line in file:
            # Split each line by tabs or spaces
            parts = line.strip().split('\t')  # You can also use split(' ') if space-separated

            # Check if there are at least two parts in the line
            if len(parts) >= 2:
                # Extract the UniProt ID from the first part
                evidence = parts[2]

                # Append the UniProt ID to the list
                evidence_list.append(evidence)

    return evidence_list


def fetch_af_models(uniprotNamesDict, className, i, j):
    uniprotIds = uniprotNamesDict[className]
    apiKey = 'AIzaSyCeurAJz7ZGjPQUtEaerUkBZ3TaBkXrY94'
    cnt = 0
    for uniprotId in uniprotIds[i:j]:
        print('i = ', i, ', j= ', j, ', cnt = ', cnt)
        cnt += 1
        api_url = f'https://alphafold.ebi.ac.uk/api/prediction/{uniprotId}?key={apiKey}'
        # Make a GET request to the AlphaFold API
        response = requests.get(api_url)
        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Parse the JSON response
            data = response.json()[0]
            # Check if .cif file URL is available
            if 'cifUrl' in data:
                # Access the .cif file URL
                cif_url = data['cifUrl']
                # Make a GET request to the .cif file URL to download it
                cif_response = requests.get(cif_url)
                # Check if the .cif file request was successful
                if cif_response.status_code == 200:
                    # Save the .cif file to a local file
                    dir_path = os.path.join(paths.GO_source_patch_to_score_path, className)
                    # check if directory exists
                    if not os.path.exists(dir_path):
                        os.makedirs(dir_path)
                    with open(os.path.join(dir_path, f'{uniprotId}.cif'), 'wb') as cif_file:
                        cif_file.write(cif_response.content)
                        print(f".cif file downloaded for {uniprotId}")
                else:
                    print(f"Error downloading .cif file: {cif_response.status_code}")
            else:
                print(f"No .cif file available for {uniprotId}")
        else:
            print(f"Error: {response.status_code} - {response.text}, {api_url}")


def is_valid_af_prediction(pdb_file_path, name):
    parser = MMCIFParser()  # create parser object
    structure = parser.get_structure(name, pdb_file_path)
    model = structure.child_list[0]
    plddtList = []
    for chain in model:
        residues = aa_out_of_chain(chain)
        for residue in residues:
            plddtList.append(residue.child_list[0].bfactor)
    above90_residues = [1 for i in range(len(plddtList)) if plddtList[i] > 90]
    if len(above90_residues) > 100 or (len(above90_residues) / len(plddtList)) > 0.2:
        return True
    return False


def uniprot_names_of_gpad_file(path):
    uniprots = []
    f = open(path, 'r')
    lines = f.readlines()
    for line in lines:
        splitted_line = line.split("\t")
        if splitted_line[0] == 'UniProtKB':
            uniprots.append(splitted_line[1])
    return uniprots


def create_uniprot_id_evidence_tuples_for_class_util(uniprot_names_dict, evidence_dict, class_name):
    names = uniprot_names_dict[class_name]
    evidences = evidence_dict[class_name]
    assert (len(evidences) == len(names))
    return [(names[i], evidences[i]) for i in range(len(evidences))]


def create_uniprot_id_evidence_tuples_for_existing_examples(uniprot_names_dict, evidence_dict):
    class_names = ['E1', 'E2', 'E3', 'ubiquitinBinding', 'DUB']
    all_tuples = []
    for class_name in class_names:
        all_tuples.extend(
            create_uniprot_id_evidence_tuples_for_class_util(uniprot_names_dict, evidence_dict, class_name))
    return all_tuples


def get_all_uniprots_for_training(all_info_dicts_path):
    all_info_dicts = load_as_pickle(all_info_dicts_path)
    uniprots = []
    for i in range(len(all_info_dicts)):
        x_cv = all_info_dicts[i]['x_cv']
        uniprots_cv = [tup[1] for tup in x_cv]
        uniprots.extend(uniprots_cv)
    return uniprots


def uniprots_evidences_list_todict(uniprot_names_evidences_list):
    uniprot_evidence_dict = dict()
    for tup in uniprot_names_evidences_list:
        uniprot_evidence_dict[tup[0]] = tup[1]
    return uniprot_evidence_dict


def get_all_valid_af_predictions_for_type(class_name):
    folder_path = os.path.join(paths.GO_source_patch_to_score_path, class_name)
    valid_list = []
    l = len(os.listdir(folder_path))
    cnt = 0
    for name in os.listdir(folder_path):
        cnt += 1
        print('length of folder is: ', l, " cnt = ", cnt)
        filePath = os.path.join(folder_path, name)
        if is_valid_af_prediction(filePath, name):
            valid_list.append(filePath)
    save_as_pickle(valid_list, os.path.join(paths.GO_source_patch_to_score_path, f'allValidOf{class_name}.txt'))


def get_uniprot_sequence_tuples_for_type(class_name):
    all_valid_paths = load_as_pickle(os.path.join(paths.GO_source_patch_to_score_path, f'allValidOf{class_name}.pkl'))
    tuples = []
    parser = MMCIFParser()  # create parser object
    for path in all_valid_paths:
        print(path)
        name = path.split("/")[-1].split(".")[-2]
        structure = parser.get_structure(name, path)
        model = structure.child_list[0]
        assert (len(model.child_list) == 1)
        seq = get_str_seq_of_chain(model.child_list[0])
        tuples.append((name, seq))
    return tuples


def create_paths_text_file_for_type(class_name):
    all_valid_paths = load_as_pickle(os.path.join(paths.GO_source_patch_to_score_path, f'allValidOf{class_name}.pkl'))
    file_path = os.path.join(paths.GO_source_patch_to_score_path, f'{class_name}Paths.txt')
    with open(file_path, 'w') as file:
        for path in all_valid_paths:
            file.write(path + '\n')


def create_all_types_list_of_sequences(list_of_name_seq_tuples):
    seq_list = []
    for list in list_of_name_seq_tuples:
        for tup in list:
            seq_list.append(tup[1])
    return seq_list


def creat_name_cluster_type_dict(cluster_indices, name_list, i, j, name_cluster_type_dict, class_name):
    assert (len(name_list) == j - i)
    for k in range(i, j):
        if name_list[k - i] in name_cluster_type_dict:
            name_cluster_type_dict[name_list[k - i]][0].append(class_name)
        else:
            name_cluster_type_dict[name_list[k - i]] = ([class_name], cluster_indices[k])


def create_name_list(name_seq_tuples):
    return [tup[0] for tup in name_seq_tuples]


def create_targets_file_for_type(class_name):
    source_directory = os.path.join(paths.GO_source_patch_to_score_path, class_name)
    output_file_path = os.path.join(paths.GO_source_patch_to_score_path, f'targets{class_name}.txt')
    with open(output_file_path, "w") as output_file:
        # Iterate over files in the source directory
        for root, dirs, files in os.walk(source_directory):
            for file in files:
                # Check if the file has a ".cif" extension
                if file.endswith(".cif"):
                    # Get the absolute path of the source file
                    source_file_path = os.path.join(root, file)
                    # Write the absolute path to the text file
                    output_file.write(source_file_path + "\n")


def split_file(input_file, output_prefix, lines_per_file=200):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    total_lines = len(lines)
    num_files = total_lines // lines_per_file + (1 if total_lines % lines_per_file != 0 else 0)

    for i in range(num_files):
        start_idx = i * lines_per_file
        end_idx = min((i + 1) * lines_per_file, total_lines)

        output_file = f"{output_prefix}_{i + 1}.txt"

        with open(output_file, 'w') as f_out:
            f_out.writelines(lines[start_idx:end_idx])


def get_all_valid_uniprot_ids_of_type(type_name, proteome=False):
    all_valid_list = load_as_pickle(os.path.join(paths.GO_source_patch_to_score_path, f'allValidOf{type_name}.pkl'))
    if proteome:
        return [path.split("/")[-1].split("-")[1] for path in all_valid_list]
    return [path.split("/")[-1][:-4] for path in all_valid_list]


def write_dict_to_csv(filename, data_dict):
    # Find the maximum length of any list in the dictionary
    max_length = max(len(v) for v in data_dict.values())

    # Fill missing values with an empty string
    for key, value in data_dict.items():
        if len(value) < max_length:
            data_dict[key].extend([''] * (max_length - len(value)))

    # Open the CSV file in write mode
    with open(filename, 'w', newline='') as csvfile:
        # Create a CSV writer object
        csv_writer = csv.DictWriter(csvfile, fieldnames=data_dict.keys())

        # Write the header row with column names
        csv_writer.writeheader()

        # Write the data rows
        for i in range(max_length):
            row = {key: data_dict[key][i] for key in data_dict}
            csv_writer.writerow(row)


def get_evidence_from_response(response):
    splitted_response = re.split(r'\t|\n', response)
    evidence_type_set = set()
    for s in splitted_response:
        if s.startswith('goEvidence'):
            evidence_type_set.add(s.split("=")[1])
    return list(evidence_type_set)


def get_evidence_of_uniprot_id(uniprot_id):
    request_url = "https://www.ebi.ac.uk/QuickGO/services/annotation/downloadSearch?includeFields=goName&selectedFields=evidenceCode&geneProductId=" + uniprot_id

    r = requests.get(request_url, headers={"Accept": "text/gpad"})
    print(uniprot_id)
    if not r.ok:
        print(uniprot_id)
        return 'request error'
    response_body = r.text
    evidence_list = get_evidence_from_response(response_body)
    evidence_list_sorted = sorted(evidence_list)
    return ",".join(evidence_list_sorted)


def update_csv_with_evidence(input_file, output_file):
    # Read the input CSV file
    with open(input_file, 'r', newline='') as infile:
        reader = csv.reader(infile)
        # Skip the header row
        header = next(reader)
        data = list(reader)

    # Process the values and add a new column

    processed_data = []
    for row in data[:5]:
        print(row)
        uniprotId = row[0]  # Get value of the first column
        evidence = get_evidence_of_uniprot_id(uniprotId)  # Process the value
        row.append(evidence)  # Add the processed value as a new column
        processed_data.append(row)

    header.append("Evidence")

    # Write the processed data to the output CSV file
    with open(output_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(header)  # Write the header row
        writer.writerows(processed_data)

    print("Processing complete. Output saved to", output_file)

