
import pandas as pd

from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger  


def generate_scaffold(smiles, include_chirality=False):
    """ Obtain Bemis-Murcko scaffold from smiles
    :return: smiles of scaffold """

    RDLogger.DisableLog('rdApp.*') #disables warnings

    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        smiles=smiles, includeChirality=include_chirality)
    return scaffold


def scaffold_split(dataset_name, frac_train=0.8):
    frac_valid = (1-0.8)/2

    # get a list of SMILES for all molecules
    path_to_smiles_mapping = f"/workspace/dataset/{dataset_name.replace('-','_')}/mapping/mol.csv.gz"

    smiles_df = pd.read_csv(path_to_smiles_mapping, compression="gzip")["smiles"]

    # create dict of the form {scaffold_i: [idx1, idx....]}
    all_scaffolds = {}
    for i, smiles in smiles_df.items():
        scaffold = generate_scaffold(smiles, include_chirality=True)
        if scaffold not in all_scaffolds:
            all_scaffolds[scaffold] = [i]
        else:
            all_scaffolds[scaffold].append(i)

    # sort from largest to smallest sets
    all_scaffolds = {key: sorted(value) for key, value in all_scaffolds.items()}
    all_scaffold_sets = [
        scaffold_set for (scaffold, scaffold_set) in sorted(
            all_scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
    ]

    # get train, valid test indices
    train_cutoff = frac_train * len(smiles_df)
    valid_cutoff = (frac_train + frac_valid) * len(smiles_df)

    train_idx, valid_idx, test_idx = [], [], []
    for scaffold_set in all_scaffold_sets:
        if len(train_idx) + len(scaffold_set) > train_cutoff:
            if len(train_idx) + len(valid_idx) + len(scaffold_set) > valid_cutoff:
                test_idx.extend(scaffold_set)
            else:
                valid_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)

    assert len(set(train_idx).intersection(set(valid_idx))) == 0
    assert len(set(test_idx).intersection(set(valid_idx))) == 0

    return {'train': sorted(train_idx), 'valid': sorted(valid_idx), 'test': sorted(test_idx)}

