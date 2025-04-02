import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
import torch
# Add the parent directory to sys.path so sibling directories can be accessed
from datasets.ogbg_dataset_extension import OGBGDatasetExtension
from datasets.qm9_dataset import QM9Dataset
from noise_experiment.feature_noise_injector import FeatureNoiseInjector
import torch, time
import numpy as np

def test_noise_distribution():
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


def test_ogb_splits():
    list_of_datasets = [
        'ogbg-molbace',
        'ogbg-molbbbp',
        'ogbg-molclintox',
        'ogbg-molmuv',
        'ogbg-molpcba',
        'ogbg-molsider',
        'ogbg-moltox21',
        'ogbg-moltoxcast',
        'ogbg-molhiv',
        'ogbg-molesol',
        'ogbg-molfreesolv',
        'ogbg-mollipo',
    ]
    for dataset_name in list_of_datasets:    
        dataset_noise = OGBGDatasetExtension(name=dataset_name, noise_level=0.1, return_types=['dgl_graph', 'targets'], device='cuda:1')
        dataset_clean = OGBGDatasetExtension(name=dataset_name, return_types=['dgl_graph', 'targets'], device='cuda:1')

        split_idx_noise = dataset_noise.get_idx_split()
        split_idx_clean = dataset_clean.get_idx_split()
        # Check if the splits are the same 
        assert np.array_equal(split_idx_noise['train'], split_idx_clean['train']), f"Train splits are not equal for {dataset_name}"
        assert np.array_equal(split_idx_noise['valid'], split_idx_clean['valid']), f"Valid splits are not equal for {dataset_name}"
        assert np.array_equal(split_idx_noise['test'], split_idx_clean['test']), f"Test splits are not equal for {dataset_name}"
        
    print("All splits are equal for all datasets")

if __name__ == '__main__':
    # noise = test_noise_distribution()
    # For freesolv we have one feature with no value to flip to therefore we will be 1/12*noise_level off
    # print(f"Fraction of noisy features: {noise}")

    test_ogb_splits()

    


    