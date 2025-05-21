import sys
import os

# from src_regression.datasets_complete_feature.molecule_datasets import MoleculeDatasetComplete
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src_regression.datasets_complete_feature.molecule_datasets import MoleculeDatasetComplete
from src_classification.datasets.molecule_datasets import MoleculeDataset
import torch
from collections import defaultdict
from tqdm import tqdm 

from noise_experiment.io import save_features, load_features

def extract_feature_values(dataset, dataset_name, save=True):
    """
    Extracts the unique values of the node and edge features of a dataset.
    Args: 
        dataset: A dataset object that has a __getitem__ method that returns a tuple with a graph and a target.
        dataset_name: str, the name of the dataset. (Used for saving)
        save: bool, whether to save the feature dictionaries to a file.
    Returns:
        seen_node_features: A dictionary with the unique values of each node feature.
        seen_edge_features: A dictionary with the unique values of each edge feature.
    """
    seen_node_features = defaultdict(set)
    seen_edge_features = defaultdict(set) 
    for i in tqdm(range(len(dataset))):
        node_features = dataset[i]['x']
        edge_features = dataset[i]['edge_attr']
        for j in range(node_features.shape[1]):
            # Get the unique values for the j-th feature
            unique_values = torch.unique(node_features[:, j])
            # Add the unique values to the dictionary
            seen_node_features[j].update(unique_values.tolist())

        for j in range(edge_features.shape[1]):
            # Get the unique values for the j-th feature
            unique_values = torch.unique(edge_features[:, j])
            # Add the unique values to the dictionary
            seen_edge_features[j].update(unique_values.tolist())   

    # Sorts the list of unique values for each feature (for better readability)
    seen_node_features = {k: sorted(list(v)) for k, v in seen_node_features.items()}
    seen_edge_features = {k: sorted(list(v)) for k, v in seen_edge_features.items()}

    # Save the features to a file. Requires a folder named 'feature_values' in the current directory.
    if save:
        save_features(seen_node_features, f'noise_experiment/feature_values/{dataset_name}_node_features.pkl')
        save_features(seen_edge_features, f'noise_experiment/feature_values/{dataset_name}_edge_features.pkl')
    return seen_node_features, seen_edge_features
    

def extract_from_all_datasets():
    dataset_regression_names = ['freesolv', 'lipophilicity', 'esol']
    dataset_classification_names = ['tox21', 'hiv', 'bace', 'bbbp', 'clintox', 'sider', 'toxcast']
    
    for name in dataset_regression_names:
        dataset_folder = 'datasets/molecule_datasets'
        dataset_folder = os.path.join(dataset_folder, name)
        dataset = MoleculeDatasetComplete(dataset_folder, dataset=name)
        extract_feature_values(dataset, name, save=True)
        print('Done with dataset:', name)
    for name in dataset_classification_names:
        dataset_folder = 'datasets/molecule_datasets'
        dataset_folder = os.path.join(dataset_folder, name)
        dataset = MoleculeDataset(dataset_folder, dataset=name)
        extract_feature_values(dataset, name, save=True)
        print('Done with dataset:', name)

if __name__ == '__main__':
    extract_from_all_datasets()
    # node_features_freesolv = load_features('noise_experiment/feature_values/hiv_node_features.pkl')
    # edge_features_freesolv = load_features('noise_experiment/feature_values/hiv_edge_features.pkl')
    # print('Node features (hiv):', node_features_freesolv)
    # print('Edge features (hiv):', edge_features_freesolv)
    extract_feature_values(MoleculeDataset("datasets/molecule_datasets/muv", dataset='muv'), 'muv', save=True)

    



