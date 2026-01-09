import pickle


def save_pickle(obj, filepath):
    """
    Save object to a pickle file.
    """
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(filepath):
    """
    Load object from a pickle file.
    """
    with open(filepath, "rb") as f:
        return pickle.load(f)
