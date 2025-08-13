import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import paths

def parse_labels(file_path):
    """
    Parses a label file into a dictionary with headers as keys and content as values.
    Args:
        file_path (str): Path to the label file.
    Returns:
        dict: A dictionary where keys are headers (without model numbers) and values are lists of info lines.
    """
    labels = {}
    with open(file_path, 'r') as file:
        current_header = None
        for line in file:
            if line.startswith('>'):
                current_header = line.strip()
                if current_header in labels:
                    raise ValueError(f"file path {file_path} Duplicate header found: {current_header}")
                labels[current_header] = []
            else:
                if current_header:
                    labels[current_header].append(line.strip())
    return labels

def aggregate_with_multiclass(multiclass_labels_path, scanNet_labels_path, output_file_path):
    """
    Aggregates labels from multiclass and ScanNet label files.

    Args:
        multiclass_labels_path (str): Path to the multiclass labels file.
        scanNet_labels_path (str): Path to the ScanNet labels file.
        output_file_path (str): Path to the output file with aggregated labels.
    """

    # Parse both label files
    multiclass_labels = parse_labels(multiclass_labels_path)
    scanNet_labels = parse_labels(scanNet_labels_path)
    multiclass_without_model = set([header[:5] + '-' + header.split('-')[1] for header in multiclass_labels.keys()])
    cnt_new = 0
    for header, content in scanNet_labels.items():
        without_model_number = header[:5] + '-' + header.split('-')[1]
        if without_model_number not in multiclass_without_model:
            cnt_new += 1
            multiclass_labels[header] = content

    # Write the aggregated labels to the output file
    with open(output_file_path, 'w') as outfile:
        for header, content in multiclass_labels.items():
            outfile.write(f"{header}\n")
            outfile.write('\n'.join(content) + '\n')
    
    print(f"Added {cnt_new} new headers from ScanNet labels to multiclass labels.")
    print(f"Aggregated labels saved to {output_file_path}")


if __name__ == "__main__":
    scanNet_labels_path = os.path.join(paths.ScanNet_Ub_PPBS_datasets, 'labels_concatenated_converted.txt')
    multiclass_labels_path = '/home/iscb/wolfson/omriyakir/other_works/PeSTo/labels_per_chain/v2/aggregated_labels_removed_duplications.txt'
    output_file_path = os.path.join(paths.ScanNet_Ub_PPBS_datasets, 'labels_aggregated_multiclass.txt')
    aggregate_with_multiclass(multiclass_labels_path, scanNet_labels_path, output_file_path)