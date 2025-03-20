import pandas as pd
import torch
import numpy as np

import os
import sys
sys.path.insert(0, 'src_classification')
sys.path.insert(1, 'src_regression')

from splitters import random_scaffold_split, random_split, scaffold_split
from datasets_complete_feature import MoleculeDatasetComplete
from torch_geometric.data import DataLoader
from config import args

if __name__ == "__main__":
    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    device = torch.device('cuda:' + str(args.device)) \
        if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)

    num_tasks = 1
    dataset_folder = 'datasets/molecule_datasets/'
    dataset_folder = os.path.join(dataset_folder, args.dataset)
    dataset = MoleculeDatasetComplete(dataset_folder, dataset=args.dataset)
    smiles_list = pd.read_csv(dataset_folder + '/processed/smiles.csv', header=None)[0].tolist()

    if args.split == 'scaffold':
        train_dataset, valid_dataset, test_dataset = scaffold_split(
            dataset, smiles_list, null_value=0, frac_train=0.8,
            frac_valid=0.1, frac_test=0.1)
        print('split via scaffold')
    elif args.split == 'random':
        train_dataset, valid_dataset, test_dataset = random_split(
            dataset, null_value=0, frac_train=0.8, frac_valid=0.1,
            frac_test=0.1, seed=args.seed)
        print('randomly split')
    elif args.split == 'random_scaffold':
        train_dataset, valid_dataset, test_dataset = random_scaffold_split(
            dataset, smiles_list, null_value=0, frac_train=0.8,
            frac_valid=0.1, frac_test=0.1, seed=args.seed)
        print('random scaffold')
    else:
        raise ValueError('Invalid split option.')
    
    test_loader = DataLoader(test_dataset)
    test_indices, test_SMILES = [], []
    for i, data in enumerate(test_loader):
        data_id = data["id"][0]
        test_indices.append(data_id)
        test_SMILES.append(smiles_list[data_id])

    print(f"\n===SMILES of molecules in test set for {args.dataset}===\n{test_SMILES}")