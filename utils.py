import os
from typing import List
import pickle


def save_as_pickle(obj: object, file_path: str) -> None:
    with open(file_path, 'wb') as file:
        pickle.dump(obj, file)


def load_as_pickle(file_path: str) -> object:
    with open(file_path, 'rb') as file:
        return pickle.load(file)
    
    
def create_paths(*paths: List[str]) -> None:
    for path in paths:
        os.makedirs(path, exist_ok=True)
