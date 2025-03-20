import pandas as pd
import sys

sys.path.insert(0, 'src_classification')
sys.path.insert(1, 'src_regression')

from splitters import scaffold_split
from datasets_complete_feature import MoleculeDatasetComplete
from torch_geometric.data import DataLoader

if __name__ == "__main__":
    task_name = "lipophilicity"
    dataset_folder = f"/workspace/datasets/molecule_datasets/{task_name}"

    dataset = MoleculeDatasetComplete(dataset_folder, dataset=f"{task_name}")
    smiles_list = pd.read_csv(dataset_folder + '/processed/smiles.csv', header=None)[0].tolist()

    train_dataset, valid_dataset, test_dataset = scaffold_split(
            dataset, smiles_list, null_value=0, frac_train=0.8,
            frac_valid=0.1, frac_test=0.1)
    
    test_loader = DataLoader(test_dataset)
    test_indices, test_SMILES = [], []
    for i, data in enumerate(test_loader):
        data_id = data["id"][0]
        test_indices.append(data_id)
        test_SMILES.append(smiles_list[data_id])

    print(f"\n===SMILES of molecules in test set for {task_name}===\n{test_SMILES}")