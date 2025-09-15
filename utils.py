import os
from typing import List
import pickle
import tensorflow as tf


def create_paths(*paths: List[str]) -> None:
    for path in paths:
        os.makedirs(path, exist_ok=True)
        

def save_as_pickle(obj: object, file_path: str, create_path: bool = True) -> None:
    if create_path:
        create_paths('/'.join(file_path.split('/')[:-1]))
    with open(file_path, 'wb') as file:
        pickle.dump(obj, file)


def load_as_pickle(file_path: str) -> object:
    with open(file_path, 'rb') as file:
        return pickle.load(file)
    
    
def save_as_tensor(data, path):
    tensor = tf.convert_to_tensor(data)
    serialized_tensor = tf.io.serialize_tensor(tensor)
    tf.io.write_file(path, serialized_tensor)


def load_as_tensor(path, out_type=tf.double):
    serialized_tensor = tf.io.read_file(path)
    tensor = tf.io.parse_tensor(serialized_tensor, out_type=out_type)
    return tensor
