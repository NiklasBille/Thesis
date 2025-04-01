import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src_classification.datasets.molecule_datasets import MoleculeDataset
from src_regression.datasets_complete_feature.molecule_datasets import MoleculeDatasetComplete


def test_regression_format():
    # Retrieve data from freesolve 
    dataset="lipophilicity"
    dataset_folder = '../datasets/molecule_datasets/'
    dataset_folder = os.path.join(dataset_folder, dataset)
    dataset = MoleculeDatasetComplete(dataset_folder, dataset=dataset, force_reload=True)

    print('dataset_folder:', dataset_folder)
    print('dataset:', dataset)
    print('len(dataset):', len(dataset))
    print('dataset[0]:', dataset[0])
    print('x tensor: ', dataset[0].x)
    print('edge_attr tensor: ', dataset[0].edge_attr)


def test_classification_format():
    # Retrieve data from tox21
    dataset="tox21"
    dataset_folder = '../datasets/molecule_datasets/'
    dataset_folder = os.path.join(dataset_folder, dataset)
    dataset = MoleculeDataset(dataset_folder, dataset=dataset)

    print('dataset_folder:', dataset_folder)
    print('dataset:', dataset)
    print('len(dataset):', len(dataset))
    print('dataset[0]:', dataset[0])
    print('x tensor: ', dataset[0].x)
    #Print edge_attr
    print('edge_attr tensor: ', dataset[0].edge_attr)

if __name__ == "__main__":
    test_regression_format()
    # test_classification_format()
