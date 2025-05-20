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
from tqdm import tqdm
from commons.utils import seed_all
def test_noise_distribution_for_dataset(name, device):
    noisy_dataset = OGBGDatasetExtension(name=name, noise_level=0.05, return_types=['dgl_graph', 'targets'], device=device)
    clean_dataset = OGBGDatasetExtension(name=name, noise_level=0.0, return_types=['dgl_graph', 'targets'], device=device)
    
    assert len(noisy_dataset) == len(clean_dataset), "Datasets should have the same length"
    total_number_of_features = 0
    num_noisy_nodes = 0
    num_noisy_edges = 0
    for i in tqdm(range(len(noisy_dataset)), desc=f"Processing {name}"):
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

def test_noise_distribution_for_all_datasets():
    list_of_datasets = [
        # 'ogbg-mollipo',
        # 'ogbg-molfreesolv',
        # 'ogbg-molesol',
        # 'ogbg-moltox21',
        # 'ogbg-molhiv',
        # 'ogbg-molbace',
        # 'ogbg-molbbbp',
        # 'ogbg-molclintox',
        'ogbg-molmuv', 
        # 'ogbg-molpcba', # Removed to make test faster 
        # 'ogbg-molsider',
        # 'ogbg-moltoxcast',
    ]
    noise_fractions = []
    for name in list_of_datasets:
        noise_fraction = test_noise_distribution_for_dataset(name, device='cuda:1')
        noise_fractions.append(noise_fraction)
    
    # Print the noise fractions for each dataset
    for name, noise_fraction in zip(list_of_datasets, noise_fractions):
        print(f"Dataset: {name}, Noise Fraction: {noise_fraction:.4f}")



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

def test_static_and_dynamic_noise():
    seed_all(0)
    # First test static noise
    dataset = OGBGDatasetExtension(name='ogbg-molfreesolv', noise_level=0.1, return_types=['dgl_graph', 'targets'], device='cuda:0', dynamic_noise=False)

    noisy_graph = dataset.__getitem__(0)[0] #[0] is to get the dgl_graph
    noisy_graph2 = dataset.__getitem__(0)[0]
    # Check if the noisy graphs are the same
    assert torch.equal(noisy_graph.ndata['feat'], noisy_graph2.ndata['feat']), "Noisy graphs are not the same"
    assert torch.equal(noisy_graph.edata['feat'], noisy_graph2.edata['feat']), "Noisy graphs are not the same"
    
    # Now test dynamic noise
    dataset = OGBGDatasetExtension(name='ogbg-molfreesolv', noise_level=0.1, return_types=['dgl_graph', 'targets'], device='cuda:0', dynamic_noise=True)
    noisy_graph = dataset.__getitem__(0)[0]
    noisy_graph2 = dataset.__getitem__(0)[0]
    # Check if the noisy graphs are different
    assert not torch.equal(noisy_graph.ndata['feat'], noisy_graph2.ndata['feat']), "Dynamic graphs are the same"
    assert not torch.equal(noisy_graph.edata['feat'], noisy_graph2.edata['feat']), "Dynamic graphs are the same"

    print("Static and dynamic noise tests passed.")

if __name__ == '__main__':
    # noise = test_noise_distribution()
    # For freesolv we have one feature with no value to flip to therefore we will be 1/12*noise_level off
    # print(f"Fraction of noisy features: {noise}")

    # test_ogb_splits()
    # test_noise_distribution_for_all_datasets()
    test_static_and_dynamic_noise()

    


    