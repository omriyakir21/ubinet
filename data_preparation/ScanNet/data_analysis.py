import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import paths

def analyze_labels(folder_path):
    label_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    total_labels = 0

    for i in range(1,6):
        file_path = os.path.join(folder_path, f'labels_fold{i}.txt')
        with open(file_path, 'r') as file:
            for line in file:
                if line.startswith('>'):
                    continue
                label = int(line.strip().split()[-1])
                if label in label_counts:
                    label_counts[label] += 1
                    total_labels += 1

    for label in label_counts:
        percentage = (label_counts[label] / total_labels) if total_labels > 0 else 0
        print(f"Percentage of label {label}: {percentage:.2f}%")

if __name__ == '__main__':
    DATE = '8_9'
    folder_path = os.path.join(paths.scanNet_data_for_training_path,f'{DATE}_dataset')
    print(f"Analyzing labels in {folder_path}")
    analyze_labels(folder_path)