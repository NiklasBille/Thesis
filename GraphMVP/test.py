import os
import torch
import numpy as np
import torch_geometric #THE ISSUE
#from torch_geometric.nn import GCNConv

if __name__ == "__main__":

    path_to_data = "./datasets/molecule_datasets/freesolv/processed/geometric_data_processed.pt"
    #data = torch.load(path_to_data)
    #cuda0 = torch.device('cuda:0')
    #tensor = torch.randn(2, 2)
    #print(tensor)
    #tensor = tensor.to(cuda0)
    #print(tensor)
    #torch.tensor([[1., -1.], [1., -1.]]).
    #print(data)