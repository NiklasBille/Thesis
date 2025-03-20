from datasets.ogbg_dataset_extension import OGBGDatasetExtension
from torch.utils.data import DataLoader, Subset
import sys
import pandas as pd


if __name__ == "__main__":
    # First get smiles strings for freesolv
    path_to_mol_info = "/workspace/dataset/ogbg_molfreesolv/mapping/mol.csv.gz"
    df = pd.read_csv(path_to_mol_info, compression="gzip")
    df_smiles = df["smiles"]
    
    # then get test set indices
    required_data = ['dgl_graph', 'targets']
    dataset = OGBGDatasetExtension(return_types=required_data, device="cpu", name="ogbg-molfreesolv")
    split_indices = dataset.get_idx_split()
    test_indices = split_indices["test"]
    test_smiles_list = df_smiles[test_indices].tolist()

    # From similar script in GraphMVP repo
    test_smiles_list_from_GraphMVP = ['c1cnc[nH]1', 'Cn1ccnc1', 'Cc1c[nH]cn1', 'CN1CCNCC1', 'CN1CCN(CC1)C', 'C1CNCCN1', 'CCNc1nc(nc(n1)Cl)NCC', 'CCNc1nc(nc(n1)SC)NC(C)C', 'CCNc1nc(nc(n1)SC)NC(C)(C)C', 'COC(=O)C1CC1', 'C1CC1', 'CC(=O)C1CC1', 'CCc1cnccn1', 'CC(C)Cc1cnccn1', 'Cc1cnccn1', 'CCOP(=S)(OCC)S[C@@H](CCl)N1C(=O)c2ccccc2C1=O', 'c1ccc2c(c1)C(=O)NC2=O', 'CC1=CC(=O)CC(C1)(C)C', 'CC1=CC(=O)[C@@H](CC1)C(C)C', 'Cn1cccc1', 'c1cc[nH]c1', 'Cc1cccs1', 'c1ccsc1', 'CC(=O)N1CCCC1', 'C1CCNC1', 'C1CCNCC1', 'CN1CCCCC1', 'CN1CCOCC1', 'C1COCCN1', 'C1COCCO1', 'C1CC=CC1', 'c1cc2ccc3cccc4c3c2c(c1)cc4', 'C1[C@@H]2[C@H]3[C@@H]([C@H]1[C@H]4[C@@H]2O4)[C@@]5(C(=C([C@]3(C5(Cl)Cl)Cl)Cl)Cl)Cl', 'C1[C@@H]2[C@H](COS(=O)O1)[C@@]3(C(=C([C@]2(C3(Cl)Cl)Cl)Cl)Cl)Cl', 'C1(C(C(C1(F)F)(F)F)(F)F)(F)F', 'CCC[N@@](CC1CC1)c2c(cc(cc2[N+](=O)[O-])C(F)(F)F)[N+](=O)[O-]', 'c1ccc2c(c1)cccn2', 'C1C=CC[C@@H]2[C@@H]1C(=O)N(C2=O)SC(Cl)(Cl)Cl', 'C[C@@H](c1cccc(c1)C(=O)c2ccccc2)C(=O)O', 'CN(C)CCC=C1c2ccccc2CCc3c1cccc3', 'c1ccc(cc1)Cn2ccnc2', 'c1ccc-2c(c1)Cc3c2cccc3', 'C=C(c1ccccc1)c2ccccc2', 'c1ccc(cc1)Oc2ccccc2', 'CN(C)CCOC(c1ccccc1)c2ccccc2', 'C1[C@H]([C@@H]2[C@H]([C@H]1Cl)[C@]3(C(=C([C@@]2(C3(Cl)Cl)Cl)Cl)Cl)Cl)Cl', 'Cc1c[nH]c2c1cccc2', 'C1CC[S+2](C1)([O-])[O-]', 'c1ccc2cc3ccccc3cc2c1', 'C1=C[C@@H]([C@@H]2[C@H]1[C@@]3(C(=C([C@]2(C3(Cl)Cl)Cl)Cl)Cl)Cl)Cl', 'C1CCC(=O)C1', 'COP(=S)(OC)SCn1c(=O)c2ccccc2nn1', 'c1(=O)[nH]c(=O)[nH]c(=O)[nH]1', 'C1=CC(=O)C=CC1=O', 'c1ccc2c(c1)ccc3c2cccc3', 'Cn1cnc2c1c(=O)n(c(=O)n2C)C', 'CC1(Cc2cccc(c2O1)OC(=O)NC)C', 'c1cc2cccc3c2c(c1)CC3', 'c1ccc2c(c1)Cc3ccccc3C2', 'C1C=CC=CC=C1', 'c1ccc(cc1)n2c(=O)c(c(cn2)N)Cl', 'C1CNC1', 'Cc1cccc(c1C)Nc2ccccc2C(=O)O', 'C1CCCC(CC1)O', 'c1ccc2c(c1)CCC2']

    common_smiles = [smiles for smiles in test_smiles_list if smiles in test_smiles_list_from_GraphMVP]
    
    print(f"Exact same set of SMILES strings: {len(common_smiles) == len(test_smiles_list) == len(test_smiles_list_from_GraphMVP)}")
