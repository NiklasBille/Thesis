import sys
import os
import torch
# Add the parent directory to sys.path so sibling directories can be accessed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets.ogbg_dataset_extension import OGBGDatasetExtension
from datasets.qm9_dataset import QM9Dataset
from noise_experiment.feature_noise_injector import FeatureNoiseInjector

def test_data_format_QM9():
    dataset = QM9Dataset(return_types=['dgl_graph', 'targets'], device='cuda')
    
    graph, target = dataset[0]  
 
    #Print each nodes features
    print(graph.ndata['feat'])

    #Print each edges features
    print(graph.edata['feat'])

def test_data_format_OGBG():
    dataset = OGBGDatasetExtension(name='ogbg-molchembl', return_types=['dgl_graph', 'targets'], device='cuda:1')
    print(len(dataset))
    # graph, target = dataset[1]  
 
    #Print each nodes features
    # print(graph.ndata['feat'])

    #Print each edges features
    # print(graph.edata['feat'])

def test_noise_injector():
    dataset = OGBGDatasetExtension(name='ogbg-molfreesolv', return_types=['dgl_graph', 'targets'], device='cuda:1')
    graph, _ = dataset[1]

    noise_injector = FeatureNoiseInjector(
        node_feature_path="noise_experiment/feature_values/ogbg-molfreesolv_node_features.pkl",
        edge_feature_path="noise_experiment/feature_values/ogbg-molfreesolv_edge_features.pkl",
        noise_probability=1,
        device=torch.device('cuda:1')
        )
    noisy_node_features = noise_injector.apply_noise(graph.ndata['feat'], feature_type='node')
    noisy_edge_features = noise_injector.apply_noise(graph.edata['feat'], feature_type='edge')
    return noisy_node_features, noisy_edge_features

if __name__ == '__main__':
    # test_data_format_QM9()
    test_data_format_OGBG()

    # noisy_node_features, noisy_edge_features = test_noise_injector()
    
    # print(noisy_node_features, noisy_edge_features)

