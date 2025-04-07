import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src_classification.datasets.molecule_datasets import MoleculeDataset
from src_regression.datasets_complete_feature.molecule_datasets import MoleculeDatasetComplete


def extract_features_regression():
    regression_datasets = ['lipophilicity', 'freesolv', 'esol']
    for dataset in regression_datasets:
        dataset_folder = 'datasets/molecule_datasets/'
        dataset_folder = os.path.join(dataset_folder, dataset)
        dataset = MoleculeDatasetComplete(dataset_folder, dataset=dataset, force_reload=True)
        print('Done with dataset:', dataset)
        print('len(dataset):', len(dataset))


def extract_features_classification():
    # Does not include pcba dataset because it is too large
    classification_datasets = ['tox21', 'hiv', 'bace', 'bbbp', 'clintox', 'muv', 'sider', 'toxcast']
    for dataset in classification_datasets:
        dataset_folder = 'datasets/molecule_datasets/'
        dataset_folder = os.path.join(dataset_folder, dataset)
        dataset = MoleculeDataset(dataset_folder, dataset=dataset, force_reload=True)
        print('Done with dataset:', dataset)
        print('len(dataset):', len(dataset))
    

if __name__ == "__main__":
    extract_features_regression()
    extract_features_classification()
    # dataset = MoleculeDatasetComplete('datasets/molecule_datasets/freesolv', dataset='freesolv', force_reload=False)
    # print("Length of freesolv:", len(dataset))
