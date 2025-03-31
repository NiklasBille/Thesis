import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
import torch
# Add the parent directory to sys.path so sibling directories can be accessed
from datasets.ogbg_dataset_extension import OGBGDatasetExtension
from datasets.qm9_dataset import QM9Dataset
from noise_experiment.feature_noise_injector import FeatureNoiseInjector
import torch, time


def test_feature_noise_injector_effectiveness():
    noisy_dataset = OGBGDatasetExtension(name='ogbg-molfreesolv', noise_level=0.10, return_types=['dgl_graph', 'targets'], device='cuda:1')
    clean_dataset = OGBGDatasetExtension(name='ogbg-molfreesolv', noise_level=0.0, return_types=['dgl_graph', 'targets'], device='cuda:1')
    
    
    assert len(noisy_dataset) == len(clean_dataset), "Datasets should have the same length"
    total_number_of_features = 0
    num_noisy_nodes = 0
    num_noisy_edges = 0
    for i in range(len(noisy_dataset)):
        noisy_graph = noisy_dataset[i][0]
        clean_graph = clean_dataset[i][0]

        noisy_node_features = noisy_graph.ndata['feat']
        noisy_edge_features = noisy_graph.edata['feat']

        clean_node_features = clean_graph.ndata['feat']
        clean_edge_features = clean_graph.edata['feat']

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



if __name__ == '__main__':
    noise = test_feature_noise_injector_effectiveness()
    # For freesolv we have one feature with no value to flip to therefore we will be 1/12*noise_level off
    print(f"Fraction of noisy features: {noise}")
    


    