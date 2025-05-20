import sys
import os
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src_classification.datasets.molecule_datasets import MoleculeDataset
from src_regression.datasets_complete_feature.molecule_datasets import MoleculeDatasetComplete
import torch

def test_noise_distribution_for_dataset(dataset_path, dataset_name, regression=True,):
    if regression:
        noisy_dataset = MoleculeDatasetComplete(dataset_path, dataset=dataset_name, noise_level=0.05, dynamic_noise=False)
        clean_dataset = MoleculeDatasetComplete(dataset_path, dataset=dataset_name)
    else:
        noisy_dataset = MoleculeDataset(dataset_path, dataset=dataset_name, noise_level=0.05, dynamic_noise=False)
        clean_dataset = MoleculeDataset(dataset_path, dataset=dataset_name)
    
    assert len(noisy_dataset) == len(clean_dataset), "Datasets should have the same length"
    total_number_of_features = 0
    num_noisy_nodes = 0
    num_noisy_edges = 0
    for i in tqdm(range(len(noisy_dataset)), desc=f"Processing {dataset_name}"):
        noisy_graph = noisy_dataset[i]
        clean_graph = clean_dataset[i]

        noisy_node_features = noisy_graph.x
        noisy_edge_features = noisy_graph.edge_attr

        clean_node_features = clean_graph.x
        clean_edge_features = clean_graph.edge_attr

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
        noise_fraction = test_noise_distribution_for_dataset(dataset_path, name, regression=True)
        noise_fractions.append(noise_fraction)
    for name in classification_datasets:
        dataset_path = os.path.join('datasets/molecule_datasets', name)
        noise_fraction = test_noise_distribution_for_dataset(dataset_path, name, regression=False)
        noise_fractions.append(noise_fraction)
    
    for name, noise_fraction in zip(regression_datasets + classification_datasets, noise_fractions):
        print(f"Dataset: {name}, Noise Fraction: {noise_fraction:.4f}")
    
def test_static_and_dynamic_noise():

    # First test for a classification dataset
    dataset_path = 'datasets/molecule_datasets/toxcast'
    dataset_name = 'toxcast'
    dataset_static = MoleculeDataset(dataset_path, dataset=dataset_name, noise_level=0.1, dynamic_noise=False)

    static_graph = dataset_static[0]
    static_graph2 = dataset_static[0]
    
    # Ensure the static graph features are the same
    assert torch.equal(static_graph.x, static_graph2.x), "Noisy graphs are not the same"
    assert torch.equal(static_graph.edge_attr, static_graph2.edge_attr), "Noisy graphs are not the same"

    dataset_dynamic = MoleculeDataset(dataset_path, dataset=dataset_name, noise_level=0.1, dynamic_noise=True)
    dynamic_graph = dataset_dynamic[0]
    dynamic_graph2 = dataset_dynamic[0]
    # Ensure the dynamic graph features are different
    assert not torch.equal(dynamic_graph.x, dynamic_graph2.x), "Dynamic graphs are the same"
    assert not torch.equal(dynamic_graph.edge_attr, dynamic_graph2.edge_attr), "Dynamic graphs are the same"

    # Now test for a regression dataset
    dataset_path = 'datasets/molecule_datasets/freesolv'
    dataset_name = 'freesolv'
    dataset_static = MoleculeDatasetComplete(dataset_path, dataset=dataset_name, noise_level=0.1, dynamic_noise=False)
    static_graph = dataset_static[0]
    static_graph2 = dataset_static[0]
    # Ensure the static graph features are the same
    assert torch.equal(static_graph.x, static_graph2.x), "Noisy graphs are not the same"
    assert torch.equal(static_graph.edge_attr, static_graph2.edge_attr), "Noisy graphs are not the same"

    dataset_dynamic = MoleculeDatasetComplete(dataset_path, dataset=dataset_name, noise_level=0.1, dynamic_noise=True)
    dynamic_graph = dataset_dynamic[0]
    dynamic_graph2 = dataset_dynamic[0]
    # Ensure the dynamic graph features are different
    assert not torch.equal(dynamic_graph.x, dynamic_graph2.x), "Dynamic graphs are the same"
    assert not torch.equal(dynamic_graph.edge_attr, dynamic_graph2.edge_attr), "Dynamic graphs are the same"
    print("Static and dynamic noise tests passed.")



if __name__ == '__main__':
    # noise_fraction = test_noise_distribution_for_dataset('datasets/molecule_datasets/toxcast', 'toxcast', regression=True)
    # print(f"Noise fraction for freesolv dataset: {noise_fraction:.4f}")
    # test_noise_distribution_for_all_datasets()
    noise_fraction = test_noise_distribution_for_dataset('datasets/molecule_datasets/muv', 'muv', regression=False)
    print(f"Noise fraction for muv dataset: {noise_fraction:.4f}")
    # test_static_and_dynamic_noise()

