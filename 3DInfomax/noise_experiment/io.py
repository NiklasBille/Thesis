import pickle
def save_features(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

def load_features(filename):
    with open(filename, 'rb') as inp:
        return pickle.load(inp)
    