import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import torch
from collections import defaultdict
from tqdm import tqdm 
from datasets.ogbg_dataset_extension import OGBGDatasetExtension
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
        graph, _ = dataset[i]
        node_features = graph.ndata['feat']
        edge_features = graph.edata['feat']
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
    

if __name__ == '__main__':
    dataset_name = 'ogbg-molmuv'

    dataset = OGBGDatasetExtension(name=dataset_name, return_types=['dgl_graph', 'targets'], device='cuda')
    
    node_dict, edge_dict = extract_feature_values(dataset=dataset, dataset_name=dataset_name, save=False)
    node_dict = load_features(f'noise_experiment/feature_values/{dataset_name}_node_features.pkl')
    edge_dict = load_features(f'noise_experiment/feature_values/{dataset_name}_edge_features.pkl')
    print("Dataset :", dataset_name)
    print(node_dict)
    print(edge_dict)
