import sys
import os

# Add the parent directory to sys.path so sibling directories can be accessed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets.ogbg_dataset_extension import OGBGDatasetExtension
from datasets.qm9_dataset import QM9Dataset

def test_data_format():
    dataset = QM9Dataset(return_types=['dgl_graph', 'targets'], device='cuda')
    
    graph, target = dataset[0]  
 
    #Print each nodes features
    print(graph.ndata['feat'])

    #Print each edges features
    print(graph.edata['feat'])

    

if __name__ == '__main__':
    test_data_format()
