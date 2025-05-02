import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np
import tensorflow as tf
from utils import save_as_pickle, load_as_pickle, create_paths


def save_as_tensor(data, path):
    tensor = tf.convert_to_tensor(data)
    serialized_tensor = tf.io.serialize_tensor(tensor)
    tf.io.write_file(path, serialized_tensor)


def load_as_tensor(path, out_type=tf.double):
    serialized_tensor = tf.io.read_file(path)
    # Adjust `out_type` as needed
    tensor = tf.io.parse_tensor(serialized_tensor, out_type=out_type)
    return tensor


def fit_protein_data(all_data_components, all_data_protein_size, all_data_number_of_components, dir_path,
                     max_number_of_components):
    # Fit the scalers
    scaler_size = StandardScaler()
    scaler_components = StandardScaler()
    scaler_size.fit(all_data_protein_size.reshape(-1, 1))
    scaler_components.fit(all_data_components)

    # Fit the encoder
    encoder = OneHotEncoder(sparse_output=False, categories=[
                            range(max_number_of_components + 1)])
    encoder.fit(all_data_number_of_components.reshape(-1, 1))

    # Save the scalers and encoder
    save_as_pickle(scaler_size, os.path.join(dir_path, 'scaler_size.pkl'))
    save_as_pickle(scaler_components, os.path.join(
        dir_path, 'scaler_components.pkl'))
    save_as_pickle(encoder, os.path.join(dir_path, 'encoder.pkl'))


def transform_protein_data(protein, scaler_size, scaler_components, encoder, max_number_of_components, with_pesto):
    scaled_size = scaler_size.transform(
        np.array([protein.size]).reshape(-1, 1))

    # Extract, scale, and pad the components
    top_components = sorted(protein.connected_components_tuples, key=lambda x: x[1]['average_scanNet_ubiq'], reverse=True)[
        :max_number_of_components]
    component_size = 9 if with_pesto else 4
    if len(top_components) == 0:
        protein_components_scaled = np.zeros((0, component_size))
    else:
        protein_components_flattend = []
        for component in top_components:
            patch_size, patch_dict = component[:2]
            patch_dict_flattened = [float(val) for val in patch_dict.values()]
            patch_flattened = [patch_size] + patch_dict_flattened
            protein_components_flattend.append(patch_flattened)
        protein_components_flattend = np.array(protein_components_flattend)
        protein_components_scaled = scaler_components.transform(
            protein_components_flattend)
    if len(protein_components_scaled) < max_number_of_components:
        padding = ((0, max_number_of_components -
                   len(protein_components_scaled)), (0, 0))
        protein_components_scaled = np.pad(
            protein_components_scaled, padding, mode='constant', constant_values=0)

    # Encode the number of components
    encoded_components = encoder.transform(
        np.array([len(top_components)]).reshape(-1, 1))

    # Convert to tensors
    scaled_size_tensor = tf.convert_to_tensor(scaled_size)
    scaled_components_tensor = tf.convert_to_tensor(protein_components_scaled)
    encoded_components_tensor = tf.convert_to_tensor(encoded_components)

    return scaled_size_tensor, scaled_components_tensor, encoded_components_tensor


def transform_protein_data_list(proteins, scaler_size_path, scaler_components_path, encoder_path,
                                max_number_of_components, with_pesto):
    scaler_size = load_as_pickle(scaler_size_path)
    scaler_components = load_as_pickle(scaler_components_path)
    encoder = load_as_pickle(encoder_path)
    scaled_sizes = []
    scaled_components_list = []
    encoded_components_list = []

    for protein in proteins:
        scaled_size_tensor, scaled_components_tensor, encoded_components_tensor = transform_protein_data(
            protein, scaler_size, scaler_components, encoder, max_number_of_components, with_pesto)

        scaled_sizes.append(scaled_size_tensor)
        scaled_components_list.append(scaled_components_tensor)
        encoded_components_list.append(encoded_components_tensor)

    # Print shapes of lists before stacking
    print("scaled_sizes list shape:", len(scaled_sizes),
          scaled_sizes[0].shape if scaled_sizes else None)
    print(f'scaled_sizes is {scaled_sizes}')
    print("scaled_components_list shape:", len(scaled_components_list),
          scaled_components_list[0].shape if scaled_components_list else None)
    print("encoded_components_list shape:", len(encoded_components_list),
          encoded_components_list[0].shape if encoded_components_list else None)
    scaled_sizes = tf.squeeze(tf.convert_to_tensor(scaled_sizes), axis=1)
    encoded_components_list = tf.squeeze(
        tf.convert_to_tensor(encoded_components_list), axis=1)
    return scaled_sizes, tf.convert_to_tensor(scaled_components_list), encoded_components_list


def normalize_and_save_data(data_for_training_folder_path, proteins, sequences, sources, uniprots, protein_paths):
    scaled_sizes, scaled_components_list, encoded_components_list = (
        transform_protein_data_list(proteins,
                                    os.path.join(
                                        scalers_folder_path, 'scaler_size.pkl'),
                                    os.path.join(
                                        scalers_folder_path, 'scaler_components.pkl'),
                                    os.path.join(
                                        scalers_folder_path, 'encoder.pkl'),
                                    MAX_NUMBER_OF_COMPONENTS, with_pesto))

    save_as_tensor(scaled_sizes, os.path.join(
        data_for_training_folder_path, 'scaled_sizes.tf'))
    save_as_tensor(scaled_components_list, os.path.join(
        data_for_training_folder_path, 'scaled_components_list.tf'))
    save_as_tensor(encoded_components_list, os.path.join(
        data_for_training_folder_path, 'encoded_components_list.tf'))
    save_as_pickle(sources, os.path.join(
        data_for_training_folder_path, 'sources.pkl'))
    save_as_pickle(uniprots, os.path.join(
        data_for_training_folder_path, 'uniprots.pkl'))
    save_as_pickle(protein_paths, os.path.join(
        data_for_training_folder_path, 'protein_paths.pkl'))
    labels = tf.convert_to_tensor(
        [0 if source in NEGATIVE_SOURCES else 1 for source in sources])
    save_as_tensor(labels, os.path.join(
        data_for_training_folder_path, 'labels.tf'))


def main(patch_to_score_model_path: str, scalers_folder_path: str):
    create_paths(scalers_folder_path)
    fit_protein_data(np.array(data_components_flattend), np.array(data_protein_size),  np.array(data_number_of_components),
                     scalers_folder_path, MAX_NUMBER_OF_COMPONENTS)
    normalize_and_save_data(data_for_training_folder_path,
                            proteins, sequences, sources, uniprots, protein_paths)
