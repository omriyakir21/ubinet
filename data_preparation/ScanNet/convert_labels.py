import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import paths

def convert_labels(pssm_file_path, output_file_path):
    """
    Converts labels in a PSSM file and writes the updated content to a new file.

    Args:
        pssm_file_path (str): Path to the input PSSM file.
        output_file_path (str): Path to the output file with converted labels.
    """
    # Define the label mapping
    label_mapping = {
        '0': '10101010101010',
        '1': '01101010101010',
        '2': '10101010101001',
        '3': '01101010101001'
    }

    with open(pssm_file_path, 'r') as infile, open(output_file_path, 'w') as outfile:
        for line in infile:
            # Check if the line is a header line
            if line.startswith('>'):
                outfile.write(line)  # Write the header line as is
            else:
                # Process info lines
                parts = line.split()
                if len(parts) > 0:
                    label = parts[-1]  # Get the last value (label)
                    if label in label_mapping:
                        parts[-1] = label_mapping[label]  # Replace the label
                    outfile.write(' '.join(parts) + '\n')  # Write the updated line

def scanNet_PSSM_files_concatenation():
    list_scanNet_datasets = [
    'train',
    'validation_70',
    'validation_homology',
    'validation_topology',
    'validation_none',
    'test_70',
    'test_homology',
    'test_topology',
    'test_none',
    ]
    list_dataset_locations = [os.path.join(paths.ScanNet_Ub_PPBS_datasets,f'labels_{dataset}.txt') for dataset in list_scanNet_datasets]
    concatenated_string = ""
    for i, file_path in enumerate(list_dataset_locations):
        if os.path.exists(file_path):  # Check if file exists
            with open(file_path, "r") as infile:
                content = infile.read().rstrip()  # Remove trailing newlines to avoid excess
                concatenated_string += content
                if i < len(list_dataset_locations) - 1:  # Ensure newline between files
                    concatenated_string += "\n"
        else:
            print(f"Warning: {file_path} not found.")
    
    # Write the concatenated string to a new file
    output_file_path = os.path.join(paths.ScanNet_Ub_PPBS_datasets, 'labels_concatenated.txt')
    with open(output_file_path, "w") as outfile:
        outfile.write(concatenated_string)
    
    return output_file_path

if __name__ == "__main__":

    plan_dict = {
        'create_scanNet_data':True,
        'Imer_path':os.path.join(paths.ImerFiles_path, 'v4'),
        'convert_scanNet_labels':True,
        'create_ubiquitin_labels':True,
                 }

    if plan_dict['create_scanNet_data']:
        scanNet_PSSM_concat = scanNet_PSSM_files_concatenation()
    if plan_dict['convert_scanNet_labels']:
        scanNet_labels_path = os.path.join(paths.ScanNet_Ub_PPBS_datasets, 'labels_concatenated.txt')
        output_labels_path = os.path.join(paths.ScanNet_Ub_PPBS_datasets, 'labels_concatenated_converted.txt')
        convert_labels(scanNet_labels_path, output_labels_path)
        print(f"Labels converted and saved to {output_labels_path}")
    if plan_dict['create_ubiquitin_labels']:
        ubiq_labels_path  = os.path.join(plan_dict['Imer_path'], 'Integrated_Checkchains_mer_filtered.txt')
        ubiq_labels_converted_path = os.path.join(plan_dict['Imer_path'], 'Integrated_Checkchains_mer_filtered_converted.txt')
        convert_labels(ubiq_labels_path, ubiq_labels_converted_path)
        print(f"Ubiquitin labels converted and saved to {ubiq_labels_converted_path}")