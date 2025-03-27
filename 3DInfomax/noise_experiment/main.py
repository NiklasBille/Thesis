import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
import torch
# Add the parent directory to sys.path so sibling directories can be accessed
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
    dataset = OGBGDatasetExtension(name='ogbg-molesol', noise_level=0.1, return_types=['dgl_graph', 'targets'], device='cuda:1')
    graph, _ = dataset[1]
if __name__ == '__main__':
    # test_data_format_QM9()
    # test_data_format_OGBG()

    test_noise_injector()
    # print(noisy_node_features, noisy_edge_features)

