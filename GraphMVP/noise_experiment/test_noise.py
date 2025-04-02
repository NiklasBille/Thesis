import sys
import os
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src_classification.datasets.molecule_datasets import MoleculeDataset
from src_regression.datasets_complete_feature.molecule_datasets import MoleculeDatasetComplete
import torch

def test_noise_distribution_for_dataset(dataset_path, dataset_name, regression=True, device='cuda:1'):
    if regression:
        noisy_dataset = MoleculeDatasetComplete(dataset_path, dataset=dataset_name, noise_level=0.05, device=device)
        clean_dataset = MoleculeDatasetComplete(dataset_path, dataset=dataset_name, device=device)
    else:
        noisy_dataset = MoleculeDataset(dataset_path, dataset=dataset_name, noise_level=0.05, device=device)
        clean_dataset = MoleculeDataset(dataset_path, dataset=dataset_name, device=device)
    
    assert len(noisy_dataset) == len(clean_dataset), "Datasets should have the same length"
    total_number_of_features = 0
    num_noisy_nodes = 0
    num_noisy_edges = 0
    for i in tqdm(range(len(noisy_dataset)), desc=f"Processing {dataset_name}"):
        noisy_graph = noisy_dataset[i]
        clean_graph = clean_dataset[i]

        noisy_node_features = noisy_graph.x
        noisy_edge_features = noisy_graph.edge_attr

        clean_node_features = clean_graph.x.to(device)
        clean_edge_features = clean_graph.edge_attr.to(device)

        num_atoms = noisy_node_features.shape[0]
        num_node_features = noisy_node_features.shape[1]
        num_edges = noisy_edge_features.shape[0]
        num_edge_features = noisy_edge_features.shape[1]

        
        num_noisy_nodes += torch.sum(noisy_node_features != clean_node_features)
        num_noisy_edges += torch.sum(noisy_edge_features != clean_edge_features)

        total_number_of_features += num_node_features * num_atoms + num_edge_features * num_edges
    total_noise = num_noisy_nodes + num_noisy_edges
    total_noise_fraction = total_noise / total_number_of_features
    return total_noise_fraction

def test_noise_distribution_for_all_datasets():
    regression_datasets = ['lipophilicity', 'freesolv', 'esol']
    # Removed muv because it is too large
    classification_datasets = ['tox21', 'hiv', 'bace', 'bbbp', 'clintox', 'sider', 'toxcast']
    noise_fractions = []
    for name in regression_datasets:
        dataset_path = os.path.join('datasets/molecule_datasets', name)
        noise_fraction = test_noise_distribution_for_dataset(dataset_path, name, regression=True, device='cuda:0')
        noise_fractions.append(noise_fraction)
    for name in classification_datasets:
        dataset_path = os.path.join('datasets/molecule_datasets', name)
        noise_fraction = test_noise_distribution_for_dataset(dataset_path, name, regression=False, device='cuda:0')
        noise_fractions.append(noise_fraction)
    
    for name, noise_fraction in zip(regression_datasets + classification_datasets, noise_fractions):
        print(f"Dataset: {name}, Noise Fraction: {noise_fraction:.4f}")
    
if __name__ == '__main__':
    # noise_fraction = test_noise_distribution_for_dataset('datasets/molecule_datasets/toxcast', 'toxcast', regression=True, device='cuda:0')
    # print(f"Noise fraction for freesolv dataset: {noise_fraction:.4f}")
    test_noise_distribution_for_all_datasets()


