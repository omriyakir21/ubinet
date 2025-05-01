import pickle


def save_as_pickle(obj: object, file_path: str) -> None:
    """
    Saves an object to a file using pickle.

    Parameters:
    obj (any): The object to be saved.
    file_path (str): The path to the file where the object should be saved.
    """
    with open(file_path, 'wb') as file:
        pickle.dump(obj, file)


def load_as_pickle(file_path: str) -> object:
    """
    Loads an object from a file using pickle.

    Parameters:
    file_path (str): The path to the file where the object is saved.

    Returns:
    any: The object loaded from the file.
    """
    with open(file_path, 'rb') as file:
        return pickle.load(file)