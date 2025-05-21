from datasets.ogbg_dataset_extension import OGBGDatasetExtension
from datasets.qm9_dataset import QM9Dataset
from train import get_arguments, parse_arguments
from commons.utils import get_random_indices, seed_all
from commons.splitters import generate_scaffold, scaffold_split, random_split
from rdkit import Chem

import sys
import os
import pandas as pd
import torch
import numpy as np
from math import floor
from tqdm import tqdm


def test_scaffold_splits_for_different_models():
    """
    Checks if the scaffold splits for a couple of datasets are the same in GraphMVH, 3DInfomax and UniMol. 
    Requires that we pass a config file.
    """
    args = get_arguments()
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.device == 'cuda' else "cpu")

    if "ogbg" in args.dataset: # These tasks have same data structure

        # First get smiles strings for the task
        path_to_mol_info = f"/workspace/dataset/{args.dataset.replace('-','_')}/mapping/mol.csv.gz"

        df = pd.read_csv(path_to_mol_info, compression="gzip")
        df_smiles = df["smiles"]

        # Then get the dataset
        full_dataset = OGBGDatasetExtension(return_types=args.required_data, device=device, name=args.dataset)

        # Finally get the SMILES strings from the other methods (see GraphMVP repo for similar script)
        if "freesolv" in args.dataset:
            test_smiles_list_from_GraphMVP = ['c1cnc[nH]1', 'Cn1ccnc1', 'Cc1c[nH]cn1', 'CN1CCNCC1', 'CN1CCN(CC1)C', 'C1CNCCN1', 'CCNc1nc(nc(n1)Cl)NCC', 'CCNc1nc(nc(n1)SC)NC(C)C', 'CCNc1nc(nc(n1)SC)NC(C)(C)C', 'COC(=O)C1CC1', 'C1CC1', 'CC(=O)C1CC1', 'CCc1cnccn1', 'CC(C)Cc1cnccn1', 'Cc1cnccn1', 'CCOP(=S)(OCC)S[C@@H](CCl)N1C(=O)c2ccccc2C1=O', 'c1ccc2c(c1)C(=O)NC2=O', 'CC1=CC(=O)CC(C1)(C)C', 'CC1=CC(=O)[C@@H](CC1)C(C)C', 'Cn1cccc1', 'c1cc[nH]c1', 'Cc1cccs1', 'c1ccsc1', 'CC(=O)N1CCCC1', 'C1CCNC1', 'C1CCNCC1', 'CN1CCCCC1', 'CN1CCOCC1', 'C1COCCN1', 'C1COCCO1', 'C1CC=CC1', 'c1cc2ccc3cccc4c3c2c(c1)cc4', 'C1[C@@H]2[C@H]3[C@@H]([C@H]1[C@H]4[C@@H]2O4)[C@@]5(C(=C([C@]3(C5(Cl)Cl)Cl)Cl)Cl)Cl', 'C1[C@@H]2[C@H](COS(=O)O1)[C@@]3(C(=C([C@]2(C3(Cl)Cl)Cl)Cl)Cl)Cl', 'C1(C(C(C1(F)F)(F)F)(F)F)(F)F', 'CCC[N@@](CC1CC1)c2c(cc(cc2[N+](=O)[O-])C(F)(F)F)[N+](=O)[O-]', 'c1ccc2c(c1)cccn2', 'C1C=CC[C@@H]2[C@@H]1C(=O)N(C2=O)SC(Cl)(Cl)Cl', 'C[C@@H](c1cccc(c1)C(=O)c2ccccc2)C(=O)O', 'CN(C)CCC=C1c2ccccc2CCc3c1cccc3', 'c1ccc(cc1)Cn2ccnc2', 'c1ccc-2c(c1)Cc3c2cccc3', 'C=C(c1ccccc1)c2ccccc2', 'c1ccc(cc1)Oc2ccccc2', 'CN(C)CCOC(c1ccccc1)c2ccccc2', 'C1[C@H]([C@@H]2[C@H]([C@H]1Cl)[C@]3(C(=C([C@@]2(C3(Cl)Cl)Cl)Cl)Cl)Cl)Cl', 'Cc1c[nH]c2c1cccc2', 'C1CC[S+2](C1)([O-])[O-]', 'c1ccc2cc3ccccc3cc2c1', 'C1=C[C@@H]([C@@H]2[C@H]1[C@@]3(C(=C([C@]2(C3(Cl)Cl)Cl)Cl)Cl)Cl)Cl', 'C1CCC(=O)C1', 'COP(=S)(OC)SCn1c(=O)c2ccccc2nn1', 'c1(=O)[nH]c(=O)[nH]c(=O)[nH]1', 'C1=CC(=O)C=CC1=O', 'c1ccc2c(c1)ccc3c2cccc3', 'Cn1cnc2c1c(=O)n(c(=O)n2C)C', 'CC1(Cc2cccc(c2O1)OC(=O)NC)C', 'c1cc2cccc3c2c(c1)CC3', 'c1ccc2c(c1)Cc3ccccc3C2', 'C1C=CC=CC=C1', 'c1ccc(cc1)n2c(=O)c(c(cn2)N)Cl', 'C1CNC1', 'Cc1cccc(c1C)Nc2ccccc2C(=O)O', 'C1CCCC(CC1)O', 'c1ccc2c(c1)CCC2']

            test_smiles_list_from_UniMol = ['c1c[nH]cn1', 'c1c[nH]cn1', 'C1CC1', 'C1CC1', 'c1cnccn1', 'c1cnccn1', 'c1cnccn1', 'O=C1NC(=O)c2ccccc21', 'O=C1NC(=O)c2ccccc21', 'O=C1C=CCCC1', 'O=C1C=CCCC1', 'c1cc[nH]c1', 'c1c[nH]cn1', 'c1cc[nH]c1', 'c1ccsc1', 'c1ccsc1', 'C1CCNC1', 'C1CCNC1', 'C1CCNCC1', 'C1CCNCC1', 'C1COCCN1', 'C1COCCN1', 'C1COCCO1', 'C1CNCCN1', 'C1=CCCC1', 'c1cc2ccc3cccc4ccc(c1)c2c34', 'C1=CC2CC1[C@@H]1[C@@H]3C[C@@H]([C@H]4O[C@@H]34)[C@H]21', 'O=S1OC[C@H]2C3C=CC(C3)[C@H]2CO1', 'C1CCC1', 'c1ccc(NCC2CC2)cc1', 'c1ccc2ncccc2c1', 'O=C1NC(=O)[C@@H]2CC=CC[C@@H]12', 'O=C(c1ccccc1)c1ccccc1', 'C=C1c2ccccc2CCc2ccccc21', 'C1CNCCN1', 'c1ccc(Cn2ccnc2)cc1', 'c1ccc2c(c1)Cc1ccccc1-2', 'C=C(c1ccccc1)c1ccccc1', 'c1ccc(Oc2ccccc2)cc1', 'c1ccc(Cc2ccccc2)cc1', 'C1=CC2CC1[C@H]1CCC[C@@H]21', 'c1ccc2[nH]ccc2c1', 'C1CC[SH2+2]C1', 'c1ccc2cc3ccccc3cc2c1', 'C1=C[C@@H]2C3C=CC(C3)[C@@H]2C1', 'C1CNCCN1', 'O=C1CCCC1', 'O=c1[nH]nnc2ccccc12', 'O=c1[nH]c(=O)[nH]c(=O)[nH]1', 'O=C1C=CC(=O)C=C1', 'c1ccc2c(c1)ccc1ccccc12', 'O=c1[nH]c(=O)c2[nH]cnc2[nH]1', 'c1ccc2c(c1)CCO2', 'c1cc2c3c(cccc3c1)CC2', 'c1ccc2c(c1)Cc1ccccc1C2', 'C1=CC=CCC=C1', 'c1ncncn1', 'O=c1cccnn1-c1ccccc1', 'C1CNC1', 'c1ccc(Nc2ccccc2)cc1', 'C1CCCCCC1', 'c1ccc2c(c1)CCC2', 'c1ncncn1', 'c1ncncn1', 'C1CC1'] 
        
        elif "lipo" in args.dataset:
            test_smiles_list_from_GraphMVP = ['Clc1cc(Nc2ncnc3[nH]nc(OCCN4CCCC4)c23)ccc1OCc5ccccn5', 'Nc1ncc([nH]1)c2ccc(F)cc2', 'O=C(NC1(CC1)C#N)[C@@H]2CCCC[C@H]2C(=O)N3CCN(CC3)c4nc5ncccc5s4', 'O=C1NC=Nc2scc(c3cccs3)c12', 'Clc1ccc(cc1)c2oc(cc2)C(=O)N(Cc3ccccn3)c4ccc(cc4)N5CCNCC5', 'CCC1=C(C)CN(C(=O)NCCc2ccc(cc2)S(=O)(=O)NC(=O)N[C@@H]3CC[C@@H](C)CC3)C1=O', 'COc1cc2ncc(C(=O)N)c(Nc3ccc(F)cc3F)c2cc1NCCN4CCCCC4', 'Oc1ccc(Nc2nc(cs2)c3ccc(cc3)C#N)cc1', 'Cn1cnc(c2ccccc2)c1C#Cc3ccnc(N)n3', 'CC1Nc2cc(Cl)c(cc2C(=O)N1c3ccccc3C)S(=O)(=O)N', 'CN(c1ccnc(Nc2cc(cc(c2)N3CCOCC3)N4CCCC4)n1)c5cc(CO)ccc5C', 'N#Cc1ccc(Nc2nccc(NC3CC3)n2)cc1', 'CN[C@@H](C)C(=O)N[C@@H](C1CCCCC1)C(=O)N[C@H]2CCCN(C2)C(=O)Cc3ccccc3', 'CN1CCOc2nc(ccc12)C#Cc3ccccc3', 'CC(=O)Nc1ccc2cnn(c3cc(NC4CC4)n5ncc(C#N)c5n3)c2c1', 'NC1CCN(CC1)c2nccc(C(=O)NCC34CC5CC(CC(C5)C3)C4)c2Cl', 'CCCCNc1nc(N)c2NC(=O)N(Cc3ccc(OCCN(C)C)nc3)c2n1', 'CCN1C=C(C=C(C)C1=O)[C@@]2(N=C(N)c3c(F)cccc23)c4cccc(c4)c5cc(ccn5)C#CC', 'Fc1ccc(cc1)n2cc(C3CCNCC3)c4cc(Cl)ccc24', 'CC1(C)CCC(=C(CN2CCN(CC2)c3ccc(cc3)C(=O)NS(=O)(=O)c4ccc(N[C@H](CCN5CCOCC5)CSc6ccccc6)c(c4)S(=O)(=O)C(F)(F)F)C1)c7ccc(Cl)cc7', 'Clc1ccc(CN2CCNCC2)cc1C(=O)NCC34CC5CC(CC(C5)C3)C4', 'CC(C)n1c(C)ncc1c2ccnc(Nc3ccc(cc3)S(=O)(=O)CC4CCCO4)n2', 'NC1(CCC1)c2ccc(cc2)N3C(=O)c4ccccc4N=C3c5ccccc5', 'O=S(=O)(Nc1nc(nn1Cc2ccccc2)c3ccccc3)c4ccccc4', 'CC(C)CN1C(=O)N(C)C(=O)c2c1sc(Cc3c[nH]c4ncccc34)c2C(=O)N5C[C@H](O)CO5', 'Oc1ccc2C[C@H]3N(CC4CC4)CC[C@@]56[C@@H](Oc1c25)c7[nH]c8ccccc8c7C[C@@]36O', 'COc1cc2ncnc(Nc3cc4ccccc4cn3)c2cc1OC', 'Nc1ncnc2c1ncn2[C@@H]3O[C@H](CSCCCNC(=O)NCc4ccccc4)[C@@H](O)[C@H]3O', 'CC(=O)Nc1nc(N)n(n1)c2ccccc2', 'OC(=O)c1cccnc1N2CCC(CN3CCC(CC3)Oc4ccc(Cl)c(Cl)c4)CC2', 'Clc1ccc2N=C3NC(=O)CN3Cc2c1Cl', 'Clc1ccc(CN2C[C@@H]3C[C@H]2CN3)cc1C(=O)NCC45CC6CC(CC(C6)C4)C5', 'CC(C)[C@H](N)C(=O)OCCOCn1cnc2C(=O)N=C(N)Nc12', 'CCOC(=O)\\C=C\\[C@H](C[C@@H]1CCNC1=O)NC(=O)[C@@H](CC(=O)[C@@H](NC(=O)c2cc(C)on2)C(C)C)Cc3ccc(F)cc3', 'CC(C)N1CCN[C@H](C1)C(=O)N2CCN(CC2)C(=O)Nc3ccc(Cl)c(Cl)c3', 'O[C@]1(CN2CCC1CC2)C#Cc3ccc(Oc4ccc(cc4)C(=O)NCc5ccnc6ccccc56)cc3', 'COc1cc2ncnc(Nc3cc(NC(=O)c4ccnc(c4)N5CCOCC5)ccc3C)c2cc1OCCN6CCOCC6', 'C[C@@H](NC1=CC(=O)CC1)c2ccc(Nc3ncc4cc(ccc4n3)c5ccncc5)cc2', 'CN1C(=O)C=C(CCc2cccc(c2)c3ccccc3)N=C1N', 'C[C@]12CCC(=O)C=C1CC[C@H]3[C@@H]4CC[C@](O)(C(=O)CO)[C@@]4(C)CC(=O)[C@H]23', 'CC[S+]([O-])c1ncccc1C2(O)CCN(CC34CC(c5ccccc35)c6ccccc46)CC2', 'CCc1cccc(c1)N(C)C(=N)Nc2cccc3ccccc23', 'CCOC(=O)[C@H](Cc1ccc(cc1)[N+](=O)[O-])NC(=O)c2ccccc2', 'Cc1sc2ncnc(SCC(=O)N3CCN(CC3)C(=O)c4occc4)c2c1C', 'CC(=O)C1=CN(Cc2ccccc2)C(=O)N(Cc3ccc(F)cc3)C1=O', 'CNC1=Nc2ncccc2C(=NC1c3cccs3)c4occc4', 'C1Cc2ncc(c3ccc4OCCOc4c3)n2C1', 'CC(C)OC(=O)C1=CN(Cc2c(F)cccc2F)c3sc(c(CN(C)Cc4ccccc4)c3C1=O)c5ccc(NC(=O)C(C)C)cc5', 'NC1(CCC1)c2ccc(cc2)c3ncc4cccnc4c3c5ccccc5', 'CCc1c(C)[nH]c2CCC(CN3CCOCC3)C(=O)c12', 'NC1(CCC1)c2ccc(cc2)c3c(ncc4nccn34)c5ccccc5', 'C1CC1Nc2ccc3nnc(c4ccccc4)n3n2', 'Cc1cc(N)c2cc(NC(=O)CCC(=O)Nc3ccc4nc(C)cc(N)c4c3)ccc2n1', 'Oc1cc(O)cc(\\C=C\\c2ccc(O)c(O)c2)c1', 'O=C(NC1=CC(=CNC1=O)c2ccncc2)[C@H](Cc3ccccc3)NC4(CC4)c5ccccn5', 'CCN(CC)C(=O)N[C@H]1C[C@H]2[C@@H](Cc3c[nH]c4cccc2c34)N(C)C1', 'CC(C)CN1C(=O)N(C)C(=O)c2c1sc(Cc3ccnc4ccccc34)c2C(=O)N5C[C@H](O)CO5', 'NC1(CCC1)c2ccc(cc2)c3c(ccn4ncnc34)c5ccccc5', 'Oc1ccc(CCNCCS(=O)(=O)CCCOCCc2oc3ccccc3c2)c4sc(O)nc14', 'COc1cc(OC)nc(n1)N2N=CC(=C(Cl)C2=O)Cl', 'CNC(=O)c1ccccc1Sc2ccc3c(\\C=C\\c4ccccn4)n[nH]c3c2', 'Cc1cc(Nc2ccccc2)n(CCC#N)n1', 'COc1cc(OC)c(cc1NC(=O)CSc2ccncc2)S(=O)(=O)N3C(C)CCc4ccccc34', 'COc1cc(F)ccc1c2cncc(CNC(=O)c3ccccc3)c2', 'O=C(N[C@H]1CN2CCC1CC2)c3ccc(s3)c4ccccn4', 'CC(C)c1ccccc1Cc2cc(C(=O)Nc3ccc(cc3)S(=O)(=O)c4ccccc4C(C)(C)C)c(O)c(O)c2O', 'COc1ccc(NC(=O)c2ccnc(N)n2)cc1', 'CC(C)(C)OC(=O)N1CCN(CC1)c2ccc(OCc3ccc(cc3)S(=O)(=O)C)cc2', 'C1CN2C[C@@H](N=C2S1)c3ccccc3', 'Nc1c(NC2CCCC2)nc(nc1N3CCOCC3)C#N', 'CC(C)CN1C(=O)N(C)C(=O)c2c1sc(Cc3ccccc3C(F)(F)F)c2C(=O)N4CCCCC4', 'C[C@H]1[C@@H]2CN(CCN3CCOCC3)CC[C@H]2Cc4[nH]c5ccc(cc5c14)C(F)(F)F', 'C[C@@H]1CN(Cc2ccc(F)cc2)[C@@H](C)CN1C(=O)c3cc4c(cn(C)c4cc3Cl)C(=O)C(=O)N(C)C', 'CCCC(=O)N1CCN(CC1)c2nnc(c3ccccc3)c4ccccc24', 'Cc1cccc(c1)N(Cc2cc(F)c(F)cc2F)C(=O)O[C@H]3CN4CCC3CC4', 'NC(=O)c1cnc(N[C@H]2CCCNC2)c3cc(sc13)c4ccncc4', 'CN1C(=O)c2c(onc2c3ccccc3)C=C1c4ccncc4', 'Cc1cc(C)c2c(N)c(sc2n1)C(=O)NC3CC3', 'COc1cccc(Nc2nc(NCC3CCCO3)c4ccccc4n2)c1', 'COc1ccc2c(C)cc(N[C@H]3CCC[C@@H](C3)NCc4cccc(OC(F)(F)F)c4)nc2c1', 'CC1(CC1)c2c(cnn2c3ccc(cc3)C(=O)O)C(=O)NC4C5CC6CC(CC4C6)C5', '[O-][N+](=O)c1ccc2c(c1)nc3CCCCCn23', 'NC1C2CN(CC12)c3nc4N(C=C(C(=O)O)C(=O)c4cc3F)c5ccc(F)cc5F', 'O[C@H](CNC(=O)C1=CNC(=O)c2cc(ccc12)S(=O)(=O)NC3CC3)CN4CCC(CC4)Oc5ccc(Cl)c(Cl)c5', 'COc1cc(ccc1Cn2ncc3ccc(NC(=O)OC4CCCC4)cc23)C(=O)NS(=O)(=O)c5ccccc5', 'CC(C)NC[C@@H](C(=O)N1CCN(CC1)c2ncnc3[C@H](O)C[C@@H](C)c23)c4ccc(Cl)cc4', 'CCCc1c(O)c(ccc1OCc2cccc(NC(=O)c3ccccc3C(=O)O)c2)C(=O)C', 'COc1ccc(C=C2SC(=O)NC2=O)cc1OC', 'Fc1ccc(cc1)C(=O)C2CCN(CCN3C(=O)Nc4ccccc4C3=O)CC2', 'Oc1ccc(CCNCCS(=O)(=O)CCCOCCSc2ccccc2)c3sc(O)nc13', 'CC(=O)CC(C1=C(O)c2ccccc2OC1=O)c3ccccc3', 'CC(C)NC(=O)c1cnc(N2CCC(CC2)N3C(=O)OCc4ccccc34)c(Cl)c1', 'NC1=NC(Nc2c(F)ccc(F)c12)c3occc3', 'Cc1ccc(NC(=O)c2cccc(c2)N3CCOCC3)cc1NC(=O)c4ccc(OCc5ccccn5)cc4', 'CC[C@@H](NC1=C(Nc2cccc(C(=O)N(C)C)c2O)C(=O)C1=O)c3ccccc3', 'CCC(C)(C)C(=O)O[C@H]1C[C@@H](C)C=C2C=C[C@H](C)[C@H](CC[C@@H]3C[C@@H](O)CC(=O)O3)[C@@H]12', 'O[C@@H](CNCCSCCCOCCc1ccccc1)c2ccc(O)c3NC(=O)Sc23', 'CS(=O)(=O)Cc1cc(nc(n1)c2cccc3cc[nH]c23)N4CCOCC4', 'C[C@@H]1CN(CCN1c2ncc(OCc3ccncc3C#N)cn2)c4ncc(F)cn4', 'CC(=O)NC[C@H]1CN(C(=O)O1)c2ccc(N3CCOCC3)c(F)c2', 'CC(C)c1nc2ccccc2n1c3nc(N4CCOCC4)c5nc(OC6CN(C6)C7CCS(=O)(=O)CC7)n(C)c5n3', 'FC(F)(F)c1nnc2ccc(NCc3cccnc3)nn12', 'COc1ccc(NC(=O)CSc2nnc(C)n2c3ccc(C)cc3)cc1', 'Cc1nc2nc(nn2c(O)c1Cc3ccccc3)c4cccnc4', 'CC(=O)NC1CC2CCCC(C1)N2C(=O)Nc3ccc(Cl)c(c3)C(F)(F)F', 'CCCCC1C(=O)N(N(C1=O)c2ccccc2)c3ccccc3', 'C[C@H](Nc1ccc2ncn(c3cc([nH]n3)C4CC4)c2n1)c5ncc(F)cn5', 'COc1cc2c(NC(=O)Nc3c(Cl)cccc3Cl)ncnc2cc1OCC4CCN(C)CC4', 'C[C@@H](O)[C@@H]1[C@H]2[C@@H](C)C(=C(N2C1=O)C(=O)OCOC(=O)C(C)(C)C)SC3CN(C3)C4=NCCS4', 'O=C(CSc1ccncc1)Nc2ccc(cc2)C(=O)c3ccccc3', 'C[C@]12CC[C@H]3[C@@H](CCc4cc(O)ccc34)[C@@H]1CCC2=O', 'C[C@]12CC[C@H]3[C@@H](CCc4cc(O)ccc34)[C@@H]1CC[C@@H]2O', 'O=C(NC1CCCCC1)c2cccnc2Sc3ccccc3', 'COc1c(N2CCO[C@@H](CN(C)C)C2)c(F)cc3C(=O)C(=CN(C4CC4)c13)C(=O)O', 'O[C@@H](CNCCCCCCCCCN1CCC(CC1)OC(=O)Nc2ccccc2c3ccccc3)c4ccc(O)c5NC(=O)C=Cc45', 'Cc1ccc(cc1)n2c(C)cc(C(=O)NS(=O)(=O)c3ccc(C)cc3)c2C', 'COc1cc2ncnc(N3CC[C@H](C3)Oc4cnc5ccccc5n4)c2cc1OC', 'Nc1ccc(cc1)S(=O)(=O)N2CCCC2', 'CC1CCN(CC1)C(=O)c2scc3OCCOc23', 'FC(F)c1nc2ccccc2n1c3nc(nc(n3)N4CCOCC4)N5CCOCC5', 'OC1(CCN(Cc2c[nH]c3ccccc23)CC1)c4ccc(Cl)cc4', 'COc1cc(c(OC)nn1)c2c(F)ccc3c(N)c(nnc23)C(=O)NC4CC4', 'Nc1nc(N)c2cc(NCc3ccc(Cl)c(Cl)c3)ccc2n1', 'CS(=O)(=O)N1CCc2c1nc(nc2c3cnc(N)nc3)N4CCOCC4', 'CC1(C)Oc2ncnc(N)c2N=C1c3ccc(cc3)[C@@H]4CC[C@@H](CC(=O)O)CC4', 'CS(=O)(=O)N1CCN(Cc2cc3nc(nc(N4CCOCC4)c3s2)c5cccc6[nH]ncc56)CC1', 'Nc1c(NC2CCOC2)nc(nc1N3CCOCC3)C#N', 'COc1cc2c(Nc3ccc(F)cc3F)c(cnc2cc1N4CCN(C)CC4)C(=O)N', 'CCc1ccc(CCOc2ccc(CC3SC(=O)NC3=O)cc2)nc1', 'NS(=O)(=O)c1ccc(CCNc2ccc3nnc(c4ccccc4)n3n2)cc1', 'NC(=N)c1ccc(cc1)C(=O)N2CCN(CC2)S(=O)(=O)c3ccc4cc(Br)ccc4c3', 'FC(F)Oc1ccc(cc1OCC2CC2)C(=O)Nc3c(Cl)cncc3Cl', 'CC(C)CNCc1ccc(cc1)c2ccccc2S(=O)(=O)N3CCCC3', 'CC(O)(C(=O)Nc1ccc2c(c1)c3ccccc3S2(=O)=O)C(F)(F)F', 'Nc1ncnc2sccc12', 'CCN(CC)c1ccc(cc1)c2nn3c(nnc3s2)c4[nH]nc5CCCc45', 'CN1C(=O)N(CC2CC2)c3nn(Cc4ccnc5ccc(Cl)cc45)c(c3C1=O)c6ncnn6C', 'CN1SC(=NC1=O)NCc2ccccc2', 'CN(C1CCN(Cc2nc3CCCCc3s2)CC1)C(=O)Cc4ccc(cc4)n5cnnn5', 'CS(=O)(=O)c1ccccc1C(=O)NCC(O)CNC2CCN(Cc3ccc(Cl)c(Cl)c3)CC2', 'COc1ccc(C=NN=Cc2ccc(OC)c(OC)c2)cc1OC', 'COc1ccc(cc1)C(=O)C2CCN(CC2)C(=O)c3occc3', 'CCN(CC)CCOc1ccc(cc1OC)N(C)C(=O)c2ccc(cc2)c3ccc(cc3)C(F)(F)F', 'NC(=N)Nc1cc(Cl)nc(n1)c2ccccc2', 'COc1ccc2c(c1)c(CC(=O)O)c(C)n2c3ccnc4cc(Cl)ccc34', 'CN[C@@H](C)C(=O)N[C@@H](C1CCCCC1)C(=O)N[C@H]2CCN(C2)C(=O)Cc3ccccc3', 'COc1cc2c(Nc3cc(CC(=O)Nc4cccc(F)c4)[nH]n3)ncnc2cc1OCCCN5CCC(CO)CC5', 'CC(C)C(NC(=O)CN1C(=O)C(=CN=C1c2ccc(F)cc2)NC(=O)OCc3ccncc3)C(=O)C(F)(F)F', 'O=C(Nc1ccccc1)c2oc3ccccc3c2', 'Nc1ccccc1NC(=O)c2ccc(cc2)N3CCOCC3', 'COC1=CC(=N/C/1=C\\c2[nH]c(C)cc2C)c3cc4ccccc4[nH]3', 'NC1(CCC1)c2ccc(cc2)c3ncc4ncccc4c3c5ccccc5', 'CC(C)C(NC(=O)CN1C(=O)C(=CN=C1C2CCCCC2)NC(=O)OCc3ccccc3)C(=O)C(F)(F)F', 'Nc1ncnc2sc(nc12)c3c(ncn3C[C@H]4CCCO4)c5ccccc5', 'COc1cc(ccc1Cn2ccc3ccc(NC(=O)OC4CCCC4)cc23)c5nn[nH]n5', 'NC12CCC(CC1)(CC2)c3ccccc3', 'CN[C@H]1CC[C@@H](c2ccc(Cl)c(Cl)c2)c3ccccc13', 'CN(c1ccnc(Nc2cc(cc(c2)N3CCN(C)CC3)N4CCOCC4)n1)c5cc(CO)ccc5C', 'CN(C)S(=O)(=O)c1ccc2Sc3ccccc3\\C(=C\\CCN4CCN(C)CC4)\\c2c1', 'Nc1n[nH]c(Nc2ccccc2)n1', 'O[C@]1(CN2CCC1CC2)C#Cc3ccc(Oc4ccc(cc4)C(=O)NC5CCCS(=O)(=O)C5)cc3', 'CNC(=O)c1c(F)cccc1Nc2nc(Nc3cc4N(CCc4cc3OC)C(=O)CN(C)C)nc5[nH]ccc25', 'CCCSc1nc(NC[C@@H]2CC[C@H](CC2)C(=O)O)ccc1C(=O)NC3CCCCC3', 'COc1cc2c(Nc3ncc(CC(=O)Nc4cccc(F)c4)s3)ncnc2cc1OCCCN5CCC(CO)CC5', 'OC(=O)c1ccc(cc1)c2ccc(Cl)c(c2)C(=O)NCC34CC5CC(CC(C5)C3)C4', 'Cc1cc(N)c2cc(NC(=O)CCCCC(=O)Nc3ccc4nc(C)cc(N)c4c3)ccc2n1', 'Nc1cc(nc2c(cnn12)C#N)c3cccs3', 'Oc1ccc(cc1)c2ccc3cc(O)ccc3c2', 'O=C(Nc1cccc2cccnc12)c3ccc(cc3)N4C(=O)[C@@H]5[C@H]6C[C@H](C=C6)[C@@H]5C4=O', 'COc1cc(ccc1Nc2ncc(Cl)c(n2)c3cnc4cc(ccn34)N5CCCC5)N6CCN(CC6)C(=O)C', 'C[C@]12Cc3cnn(c3C=C1CC[C@@]2(O)CCc4ccc(F)cc4C(=O)N)c5ccc(F)cc5', 'Oc1ccc(cc1)c2csc(n2)c3ccc(O)cc3', 'CC(C)(C)c1cc(O)n2nc(N)nc2n1', 'COc1cc(C)c(c(C)c1)S(=O)(=O)N(C)CCOCC(=O)N2CCN(CC2)C3CCN(C)CC3', 'C[C@H](C1=CNC(=S)N1)c2cccc(C)c2C', 'COc1ccc(N(C(C(=O)NC2CCCC2)c3ccccc3F)C(=O)c4occc4)c(OC)c1', 'CN(c1ccnc(Nc2cc(CN3CCOCC3)cc(c2)N4CCOCC4)n1)c5cc(CO)ccc5C', 'Nc1ccccc1NC(=O)c2ccc(cc2)C3CCN(CCC(=O)Nc4ccccc4F)CC3', 'Clc1cccc(\\C=C\\2/SC(=O)NC2=O)c1N3CCNCC3', 'C[C@@H](NC1=CC(=O)NCC1)c2ccc(Nc3ncc4cc(ccc4n3)c5ccncc5)cc2', 'CN1CCN(CC1)c2nc(CCOc3ccc(C[C@H](Nc4ccccc4C(=O)c5ccccc5)C(=O)O)cc3)c(C)s2', 'Cc1onc(n1)c2ccc(OCC3CN(C3)c4ccc(C)nn4)cc2', 'O=C(Nc1nnn[nH]1)C2=CC(=C3C=CC=CN3C2=O)Oc4ccccc4', 'CC(=O)Nc1ccc2ccn(c3cc(Nc4ccn(C)n4)n5ncc(C#N)c5n3)c2c1', 'FC(F)(F)COc1ccc(OCC(F)(F)F)c(c1)C(=O)NCC2CCCCN2', 'CC1(C)[C@@H]2CC[C@@]1(CS(=O)(=O)NC3CCN(CC3)c4ccc(cn4)C(F)(F)F)C(=O)C2', 'CCCN(c1cccnc1)P(=O)(c2ccccc2)c3ccccc3', 'NS(=O)(=O)c1ccc(NCc2ccccc2)cc1', 'O[C@@H](CNCCc1cccc(CN2CCC(CC2)c3ccccc3)c1)c4ccc(O)c5NC(=O)Sc45', 'CN1[C@@H]2CC[C@H]1C[C@H](C2)OC(c3ccccc3)c4ccccc4', 'OC1=CC=Cc2cc(O)c(O)c(O)c2C1=O', 'CCOc1ccc(Oc2ccc(cc2)S(=O)(=O)NC(=O)c3cccc(c3)c4ccc(Cl)c(Cl)c4)cc1', 'CCCc1c(OCCCSc2ccc(CC(=O)O)cc2Cl)ccc3c(CC)noc13', 'CN[C@@H](C)C(=O)N[C@@H](C1CCCCC1)C(=O)N[C@H]2C[C@@H]3CC[C@H]2N(CCc4ccccc4)C3', 'FC(F)(F)Oc1ccc(CNC(=O)C2N(CC3CC(F)(F)C3)C(=O)c4ccccc24)cc1', 'CC(CCc1cccc(OCc2ccc3ccccc3n2)c1)CC(=O)O', 'Oc1ccc(cc1)c2cnc3ccccc3n2', 'CN(c1ccnc(Nc2cc(cc(c2)N3CCOCC3)N4CCCCC4)n1)c5cc(CO)ccc5C', 'OC(=O)c1cc(nc2ccc(F)cc12)c3ccncc3', 'CN1CCCC1c2cccnc2', 'CNC(=O)c1ccc(CCC(COc2ccc(cc2)c3cccc(c3)[N+](=O)[O-])N4C(=O)CSC4=O)cc1', 'COC1(CCOCC1)c2ccc(NC(=O)C3=CC(=O)c4cc(F)cc(c4O3)c5c(C)nn(C)c5C)cn2', 'CCCS(=O)(=O)N1CCCC(C1)C(=O)N2CCC(Cc3ccccc3)CC2', 'COc1cccc(c1)C(=O)N2CCC(CC2)N3C(=O)Nc4ccccc34', 'CN1C(=N)N(CC(=O)c2ccc(Cl)cc2)c3ccccc13', 'O=C(NC1CCCC1)C(N(Cc2occc2)C(=O)c3ccc([nH]3)c4ccccc4)c5ccncc5', 'O=C(N1CCCC1)c2ccc(cc2)C(=C3CCN(Cc4cscn4)CC3)c5cccc6cccnc56', 'COc1cc(N2CC[C@H](C2)N(C)C)c3NC(=CC(=O)c3c1)C(=O)Nc4ccc(cc4)N5CCOCC5', 'CC(C)c1ccccc1Nc2nc(SCc3cccs3)n[nH]2', 'COc1ccc2nc(C)cc(N3CC(CNC(=O)C4CC4)OC3=O)c2c1', 'Fc1cc(F)cc(c1)c2cc(on2)N(CCCN3CCCCCC3)Cc4ccc5OCOc5c4', 'CC(=O)Nc1ccc2CCN(c3cc(NC4CC4)n5ncc(C#N)c5n3)c2c1', 'C[C@@H]1CN(CCN1C(=O)[C@@H]2CCCC[C@H]2C(=O)NC3(CC3)C#N)c4ccc5c(C)nnc(C)c5c4', 'NC(=O)Nc1sc(cc1C(=O)N)c2ccc(CNC3CCCC3)cc2', 'CC(=O)Nc1ccc2ccn(c3cc(NC4COC4)n5ncc(C#N)c5n3)c2c1', 'COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1NC(=O)\\C=C\\CN4CCCCC4', 'CC[C@H]1OC(=O)[C@H](C)[C@@H](O[C@H]2C[C@@](C)(OC)[C@@H](O)[C@H](C)O2)[C@H](C)[C@@H](O[C@@H]3O[C@H](C)C[C@@H]([C@H]3O)N(C)C)[C@](C)(O)C[C@@H](C)\\C(=N/OCOCCOC)\\[C@H](C)[C@@H](O)[C@]1(C)O', 'O[C@@H](CNCCc1cccc(CN2CCCC2)c1)c3ccc(O)c4NC(=O)Sc34', 'CO[C@@H]1CC[C@@]2(CC1)Cc3ccc(cc3C24N=C(C)C(=N4)N)c5cncc(Br)c5', 'COc1ccc(CNC(=O)Nc2ncc(s2)[N+](=O)[O-])cc1', 'O=C(NC1(CC1)C#N)[C@@H]2CCCC[C@H]2C(=O)N3CCc4oc5ccccc5c4C3', '[O-]c1c2c(nn1c3ccc(Cl)cc3)c4ccccc4c[n+]2Cc5ccccc5', 'CC(C)(C)c1cc2nc3C(=O)N(Cc4ccccc4)Cc3c(O)n2n1', 'O=C1C=C(Oc2c1cccc2c3cccc(c3)c4ccsc4)N5CCOCC5', 'CS(=O)(=O)c1ccc2nc(sc2c1)N3CCN(CC3)C(=O)[C@@H]4CCCC[C@H]4C(=O)NC5(CC5)C#N', 'CC(C)C(NC(=O)CN1C(=O)C(=CN=C1c2cccs2)NC(=O)OCc3ccccc3)C(=O)C(F)(F)F', 'NC(=O)Nc1sc(cc1C(=O)N)c2ccc(OCCN3CCCCC3)cc2', 'O=C(N1CCCCC1)c2scc3CCCCc23', 'COc1ccc2N=CC(=O)N(CCN3CCC(CC3)NCc4cc5OCCOc5cn4)c2n1', 'CC(=O)c1ncccc1NC(=O)[C@@H]2CC[C@H](CC2)N3C(=O)[C@@H]4[C@@H]5CC[C@@H](C5)[C@@H]4C3=O', 'Clc1cccc(c1)N2CCN(CCCN3N=C4C=CC=CN4C3=O)CC2', 'CC(C)C(NC(=O)CN1C(=O)C(=CC=C1c2ccccc2)NC(=O)CN3CCOCC3)C(=O)C(F)(F)F', 'CN1CCOc2cc(COC3CCCCC3)cnc12', 'CC(C)(F)C[C@H](N[C@@H](c1ccc(cc1)c2ccc(cc2)S(=O)(=O)C)C(F)(F)F)C(=O)NC3(CC3)C#N', 'O[C@H](CNC(=O)N1C(=O)Nc2ccccc12)CN3CCC(CC3)Oc4ccc(Cl)c(Cl)c4', 'C[C@@H](O)[C@H](NC(=O)c1ccc(cc1)C#Cc2ccc(CN3CCOCC3)cc2)C(=O)NO', 'Cc1c(Sc2ccc(Cl)cc2)c3c(Cl)nccc3n1CC(=O)O', 'CN[C@@H](C)C(=O)N[C@@H](C1CCCCC1)C(=O)N[C@H]2CCN(CCc3ccccc3)C2', 'Clc1ccc(cc1)C(=O)C2CCN(CC2)C(=O)C3CC3', 'OC(=O)c1ccccc1Cn2nnc(n2)c3cccc(OCc4ccc5ccccc5n4)c3', 'Cc1cnc(cn1)C(=O)NCCc2ccc(cc2)S(=O)(=O)NC(=O)NC3CCCCC3', 'CC(C)Oc1cc(n[nH]1)n2cnc3cnc(N[C@@H](C)c4ncc(F)cn4)nc23', 'C[C@]1(CCSC(=N1)N)c2cc(c(F)cc2F)c3cncnc3', 'CN1CCCC(CN2CCN(CC2)C(=O)Nc3ccc(Cl)c(Cl)c3)C1', 'Cc1ccc(cc1)S(=O)(=O)Nc2c(cnn2c3ccccc3)C(=O)NCc4ccccc4', 'c1ccc(cc1)c2nnc(s2)c3ccccc3', 'O=C(N1CCN(CC1)c2ncccn2)c3cccc(CC4=NNC(=O)c5ccccc45)c3', 'Nc1nc(OCc2nccs2)nc3c1ncn3[C@@H]4O[C@H](CF)[C@@H](O)[C@H]4O', 'CCCN[C@H]1CCc2nc(N)sc2C1', 'Clc1ccc(O[C@H]2CCCNC2)cc1C(=O)NCC34CC5CC(CC(C5)C3)C4', 'C[C@@H](CNC(=O)c1c(O)c(O)cc2c(O)c(c(C)cc12)c3c(C)cc4c(C(=O)NC[C@H](C)c5ccccc5)c(O)c(O)cc4c3O)c6ccccc6', 'Clc1ccc(NS(=O)(=O)c2ccc(Cl)s2)c(c1)C(=O)Nc3ccc(cc3)S(=O)(=O)N4CCOCC4', 'CC(C)n1ncc2c(cc(nc12)C3CC3)C(=O)NCc4c(C)cc(C)nc4O', 'CC1(C)N(Cc2ccnc(c2)N3CCOCC3)C(=O)N(C1=O)c4ccc(SC(F)(F)F)cc4', 'CCOC(=O)C1(CCCN(CC2CC2)C1)c3cccc(O)c3', 'Cn1c(nc2ccccc12)c3ccc(O)nc3', 'CC(C)Oc1cc(OCCc2cccnc2)cc(c1)C(=O)Nc3ccc(cn3)C(=O)O', 'Cc1ccc(cc1NC(=O)c2cnn(c2N)c3ccccc3F)C(=O)Nc4ccon4', 'CCN(CC)S(=O)(=O)c1ccc(cc1)c2oc(SCC(=O)N3CCc4ccccc34)nn2', 'C[C@@H]1CN(CCN1c2ncc(OCc3ccncc3C#N)cn2)c4noc(n4)C5CC5', 'CS(=O)(=O)Cc1cc(nc(n1)c2cnc3[nH]ccc3c2)N4CCOCC4', 'C(CNCc1ccccc1)NCc2ccccc2', 'NS(=O)(=O)c1cc2c(NC(Cc3ccccc3)NS2(=O)=O)cc1C(F)(F)F', 'COc1cc2ncnc(Nc3cc(NC(=O)c4ccnc(c4)N5CCOCC5)ccc3C)c2cc1OCCN6CCCC6', 'O=C1CCOc2nc(\\C=C\\c3ccccc3)ccc12', 'CC(C)N1CCN(Cc2oc(nc2)c3cc(cc4[nH]ncc34)c5cccc6[nH]ccc56)CC1', 'CC1N(C(=O)c2ccccc2)c3ccccc3NC1=O', 'O[C@H](CNC(=O)c1n[nH]c2ccccc12)CN3CCC(CC3)Oc4ccc(Cl)c(Cl)c4', 'NC1(CCN(CC1)c2ncnc3[nH]ccc23)C(=O)NC(C4CC4)c5ccc(Cl)cc5', 'CON(C)C(=O)c1c(Cn2c(C)nc(Cl)c2Cl)sc3N(CC(C)C)C(=O)N(C)C(=O)c13', 'CN1CCN(CC1)c2ccc(NC(=O)c3oc(Nc4ccccc4F)nn3)cn2', 'Nc1ncc(nc1C(=O)Nc2cccnc2)c3ccc(cc3)S(=O)(=O)N4CCOCC4', 'CCOC(=O)C(CCc1ccccc1)NC2CCc3ccccc3N(CC(=O)O)C2=O', 'CCOc1ccc2oc(C(=O)NC(CCSC)c3nc4ncccc4[nH]3)c(C)c2c1', 'Nc1nnc(c(N)n1)c2cccc(Cl)c2Cl', 'Clc1ccc(cc1)C2(CCCC2)C(=O)Nc3oc(nn3)C(=O)Nc4ccc(cc4)N5CCOCC5', 'Fc1ccc(CN2CCN(CC2)C3=Nn4c(CC3)nnc4C(F)(F)F)cc1', 'CC[C@@H](NC1=C(Nc2cccc(C(=O)N(C)C)c2O)C(=O)C1=O)c3oc(C)cc3', 'COc1ccc(cc1)C(=O)NCc2cn(nn2)c3cc(C)nc4ccc(OC)cc34', 'Nc1nc(N)c2nc(c(N)nc2n1)c3ccccc3', 'CC(C)C(=O)Nc1nc(cc2ccccc12)c3ccccn3', 'COc1cc2ncn(c3cc(OCc4ccccc4C(F)(F)F)c(s3)C(=O)O)c2cc1OC', 'CC1CN(C(=O)c2ccccc2)c3ccccc3NC1=O', 'C[C@]1(CCCN1c2nc(Nc3cc([nH]n3)C4CC4)c5cccn5n2)C(=O)Nc6ccc(F)nc6', 'CC(C)(C)NS(=O)(=O)c1cncc(c1)c2ccc3nc(NC(=O)NCC(=O)N4CCOCC4)nn3c2', 'COc1ccccc1c2nnc(N)[nH]2', 'CN1CCN2C(C1)c3ccccc3Cc4cccnc24', 'O=C1CCOc2nc(ccc12)C#CC3CCCC3', 'Cc1ccc(F)cc1S(=O)(=O)N[C@@H]2CCN(Cc3ccc(cc3)c4ccccc4)C2', 'NC(=O)N1c2ccccc2CC(=O)c3ccccc13', 'CC1CC(=O)Nc2ccccc2N1', 'Cc1cc(C(=O)CN2C=C(C=CC2=O)C(F)(F)F)c(C)n1Cc3ccccc3', 'COc1ccc(cc1)c2cc(Nc3ccn(C)n3)n4ncc(C#N)c4n2', 'O=C1COC2(CCN(CC2)S(=O)(=O)c3ccc(cc3)c4ccc5cnccc5c4)CN1C6CC6', 'N#CC[C@H](C1CCCC1)n2cc(cn2)c3ncnc4[nH]ccc34', 'CC(C)C(NC(=O)CN1C(=O)C(=CC=C1c2cccnc2)NC(=O)OCc3ccccc3)C(=O)C(F)(F)F', '[O-][N+](=O)c1cccc(c1)C(=O)Nc2nc3ccccc3n2CCN4CCOCC4', 'CSCCC(NC(=O)c1sccc1Cl)c2nc3ccccc3[nH]2', 'NC1(CCC1)c2ccc(cc2)c3nn4cccc4cc3c5ccccc5', 'Clc1ccc2OC(=O)Nc2c1', 'O=C1CCOc2nc(CCc3ccccc3)ccc12', 'CS(=O)(=O)c1ccc(cc1)c2cnc(N)c(c2)c3ccc(nc3)C(F)(F)F', 'COc1ccc2ncc(F)c(CCN3CCC(CC3)NCc4cc5OCCOc5cn4)c2n1', 'Fc1ccc(SC2=NN3C=NC(=O)C(=C3C=C2)c4c(Cl)cccc4Cl)c(F)c1', 'C[C@H](NC(=O)c1c(C)nn(C2CCOC2)c1NS(=O)(=O)c3ccc(C)cc3)C(C)(C)C', 'O=C1NC=Nc2[nH]ncc12', 'Cc1ccccc1N2C(=Nc3cccc(C)c3C2=O)Cn4nc(c5ccc(O)c(F)c5)c6c(N)ncnc46', 'O=C(C1CCCCC1)N2CC3N(CCc4ccccc34)C(=O)C2', 'Cc1nnsc1C(=O)Nc2ccccc2', 'O=C(COc1ccccc1)N2CCOCC2', 'CN1CCN(CC1)c2ncnc3c2sc4nc(C)cc(C)c34', 'COc1cccc([C@H](O)C2CCN(CCc3ccc(F)cc3)CC2)c1OC', 'O=C(Nc1nc2C(=O)NC=Nc2s1)c3ccccc3', 'C(C(C1CCCCC1)C2CCCCC2)C3CCCCN3', 'CCCCc1ncc(\\C=C(/Cc2cccs2)\\C(=O)O)n1Cc3ccc(cc3)C(=O)O', 'Oc1c2C(=O)N(NC(=O)c2nc3cc(Cl)ccc13)C(C4CC4)c5ccncc5', 'CN(C)c1nc(N[C@@H]2CC[C@@H](CC2)NC(=O)c3ccc(F)c(F)c3)nc4ccccc14', 'COc1ccccc1C(NC(=O)[C@@H]2CCCC[C@H]2C(=O)N3CCN(Cc4ccc(F)cc4)CC3)C#N', 'Nc1ccccc1NC(=O)c2ccc(cc2)C3CCN(Cc4ccc(cc4)C(=O)NC5CC5)CC3', 'C(CN1CCCCC1)C2CCc3cc(OCc4ccc(cc4)c5ccccc5)ccc3C2', 'CC(C)CN1C(=O)N(C)C(=O)c2c1sc(Cc3c[nH]c4ccccc34)c2C(=O)N5CC[C@@H](O)C5', 'O=C(NCCSCc1ccccc1)c2ccccc2', 'CN[C@@H](C)C(=O)N[C@@H](C1CCCCC1)C(=O)N[C@H]2CCN(C2)C(=O)c3ccccc3', 'CCCCCC[C@H]1OC(=O)CNC(=O)[C@H](NC(=O)[C@H](CO)NC(=O)[C@@H](NC(=O)[C@H](CC(C)C)N(C)C(=O)[C@@H]1C)[C@H](C)CC)[C@H](C)O', 'CN(C1CCN(C)CC1)c2nccc(Nc3cc(NC(=O)c4ccnc(c4)N5CCOCC5)ccc3C)n2', 'Cc1ccc2c(n1)c(Sc3ccc(Cl)cc3)c(C)n2CC(=O)O', 'Clc1ccc(CNC(=N)SCCCc2c[nH]cn2)cc1', 'OC(=O)C1CCN(CC1)c2ncccc2c3ccc(Cl)c(c3)C(=O)NCC4CCCCCC4', 'CC(C)CN1C(=O)N(C)C(=O)c2c1sc(Cc3cccc4ncccc34)c2C(=O)N5CC[C@@H](O)C5', 'C[C@@H](NC(=O)[C@H]1CCCCN1)c2ccc(Nc3ncc4cc(ccc4n3)c5ccncc5)cc2', 'Clc1ccc2C(=O)C3=C(Nc2c1)C(=O)NN(Cc4ccccn4)C3=O', 'CN1CCCN(CC1)C(c2ccccc2)c3ccc(Cl)cc3', 'Clc1cccc(c1Cl)n2nnnc2NCc3ccccc3Oc4ccccn4', 'OC(=O)COc1ccc(Cl)cc1CN2CCCN(CC2)S(=O)(=O)Cc3ccccc3', 'Cc1cccc(CN2CCN(CC2)C(c3ccccc3)c4ccc(Cl)cc4)c1', 'C[C@@H](NCc1ccccc1c2ccc(CCNC[C@H](O)c3ccc(O)c4NC(=O)Sc34)cc2)c5ccccc5', 'Cc1ccccc1c2c(C(=O)O)n(CCCOc3cccc4ccccc34)c5ccccc25', 'CC(CN1CCCCC1)C(=O)c2ccc(C)cc2', 'Fc1ccc(NC(=O)c2cccnc2Oc3ccccc3)cc1', 'Clc1ccc(CC2CCNCC2)cc1C(=O)NCC34CC5CC(CC(C5)C3)C4', 'C[C@@H]1CN([C@@H](C)CN1C(=O)c2ccc(cc2)C#N)C(=O)[C@@](C)(O)C(F)(F)F', 'NC1(CCC1)c2ccc(cc2)c3ncc4ccncc4c3c5ccccc5', 'CN(C)C(=O)CN1CCN(CCc2c([nH]c3sc(cc23)C(C)(C)C(=O)N4C5CCC4CC5)c6cc(C)cc(C)c6)CC1', 'C[C@@H](NC(=O)C1CCCCN1C)c2ccc(Nc3ncc4cc(ccc4n3)c5ccncc5)cc2', 'CC1CC(CC(C)(C)C1)OC(=O)C(O)c2ccccc2', 'CCOc1ncc(C)c2NC(=C([C@@H](c3ccc(cc3OC)C#N)c12)C(=O)N)C', 'CC(=O)Nc1nc(C)c(s1)c2cnc(F)c(NS(=O)(=O)c3sc(C)nc3C)c2', 'COc1cc2c(Nc3ccc(Cl)cc3F)ncnc2cc1OCCn4cncn4', 'COc1ccc(Cn2cc3N(CC(C)C)C(=O)N(C)C(=O)c3c2)cc1', 'C[C@@H](Oc1cccc2ncnc(Nc3ccc4c(cnn4Cc5ccccn5)c3)c12)C(=O)N6CCOCC6', 'CCCNC(=O)CSc1ccc(cn1)S(=O)(=O)N2CCCC2', 'FC(F)(F)c1nnc2ccc(nn12)N3CCCCC3', 'O=C(Oc1ccccc1)N2CCOCC2', 'Clc1ccc(cc1C(=O)NCC2CCCCC2)N3N=CC(=O)NC3=O', 'CC(C)CN1C(=O)N(C)C(=O)c2c1sc(Cn3c(C)nc(Cl)c3Cl)c2C(=O)N4CC[C@@H](O)C4', 'CC1CCc2cc(F)cc3C(=O)C(=CN1c23)C(=O)O', 'COc1cccc(NC(=O)Cn2cc(Oc3ncnc4cc(OC)c(OC)cc34)cn2)c1', 'NC(=O)c1ccc(Oc2cccc3cccnc23)c(c1)[N+](=O)[O-]', 'Clc1ccccc1c2cnn[nH]2', 'OCC(CO)N1C=Cc2c(NC(=O)CC34CC5CC(CC(C5)C3)C4)cccc2C1=O', 'O[C@@H](CNCCc1ccccc1CNCCc2ccccc2F)c3ccc(O)c4NC(=O)Sc34', 'Clc1ccc(N2CCN(CC2)C(=O)COCc3cscn3)c(Cl)c1', 'Oc1nc(nc2CCSCc12)c3ccc(cc3)C(F)(F)F', 'Oc1ccc2C(=O)C=C(Oc2c1)N3CCOCC3', 'COC1\\C=C\\OC2(C)Oc3c(C)c(O)c4C(=O)C(=CC(=O)c4c3C2=O)NC(=O)\\C(=C/C=C/C(C)C(O)C(C)C(O)C(C)C(OC(=O)C)C1C)\\C', 'Nc1nnc(CCSCCc2nnc(NC(=O)Cc3ccccc3)s2)s1', '[O-][N+](=O)c1ccc2OC(CN(c2c1)c3cccc[n+]3[O-])(C(F)F)C(F)F', '[O-][S+](c1ccccc1)c2ccc3nnnn3n2', 'O=C1C=COc2cc(OCc3ccccc3)ccc12', 'Cn1ncnc1COc2nn3c(nnc3cc2C(C)(C)C)c4cc(F)ccc4F', 'CN1CCN(CC1)C2=Nc3cc(Cl)ccc3Nc4ccccc24', 'Nc1c(NC2CC3CCC2C3)nc(nc1N4CCOCC4)C#N', 'NC(=N)NC(=O)c1nc(Cl)c(NCc2ccccn2)nc1N', 'CN(C1CCN(CCC(c2ccccc2)c3ccccc3)CC1)C(=O)C4CCC4', 'CC(C(Cc1nc2ccccc2[nH]1)c3nc4ccccc4[nH]3)c5ccccc5', 'NC1(CCC1)c2ccc(cc2)c3nc4C=CN5C(=O)NN=C5c4cc3c6ccccc6', 'CC(C)C[C@H](N)c1oc(nn1)S(=O)(=O)Cc2ccc(F)cc2', 'Nc1c2CCN(c3ccccc3)c2nc4ccc(Br)cc14', 'Oc1ccccc1OC(=O)c2cccc3ccccc23', 'O[C@@H](CNC1CCN(Cc2ccc(Cl)c(Cl)c2)CC1)COc3cccc(c3)C#N', 'C1Oc2ccccc2C3=NN(CC13)c4ccccc4', 'COc1ccc(Cc2c(N)n[nH]c2N)cc1', 'CCN(CC)CCCCNc1ncc2CN(C(=O)N(Cc3cccc(NC(=O)C=C)c3)c2n1)c4c(Cl)c(OC)cc(OC)c4Cl', 'CC(C)N(CCCNC(=O)Nc1ccc(cc1)C(C)(C)C)C[C@H]2O[C@H]([C@H](O)[C@@H]2O)n3cnc4c(N)ncnc34', 'CCN1CCN(CC1)c2ccc(Nc3cc(ncn3)N(C)C(=O)Nc4c(Cl)c(OC)cc(OC)c4Cl)cc2', 'COC(=O)[C@H]1[C@@H](O)CC[C@H]2CN3CCc4c([nH]c5ccccc45)[C@@H]3C[C@H]12', 'Cc1cc(CCCOc2c(Cl)cc(cc2Cl)C3=NCCO3)on1', 'CNCCCC12CCC(c3ccccc13)c4ccccc24', 'COC(=O)c1ccc(C)c(NS(=O)(=O)c2ccc3N(C)SC(=O)c3c2)c1', 'CC1(CC1)c2nc(ncc2C(=O)N[C@@H]3C4CC5CC3C[C@@](O)(C5)C4)N6CCOCC6', 'C[C@H](Nc1ncc(F)c(Nc2cc([nH]n2)C3CC3)n1)c4ncc(F)cn4', 'Cc1cc(CCC2CCN(CC2)S(=O)(=O)CC3(CCOCC3)N(O)C=O)c(C)cn1', 'Clc1ccc(CN2CC3CNCC(C2)O3)cc1C(=O)NCC45CC6CC(CC(C6)C4)C5', 'O=C1CCOc2cc(COc3ccccc3)ccc12', 'COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN4CCCC4', 'CO[C@@H]1CC[C@@]2(CC1)Cc3ccc(OCC(C)C)cc3C24N=C(C)C(=N4)N', 'O=C(COc1ccccc1)c2ccccc2', 'Cc1ccccc1NC(=O)CCS(=O)(=O)c2ccc(Br)s2', 'OC(C(=O)OC1CN2CCC1CC2)(c3ccccc3)c4ccccc4', 'O=C1NC(=NC(=C1C#N)c2ccccc2)SCCc3ccccc3', 'N(c1ccccc1)c2cc(Nc3ccccc3)[nH]n2', 'C[C@H](NC(=O)c1c(C)nn(C2CCCC2)c1NS(=O)(=O)c3ccc(C)cc3)C(C)(C)C', 'COc1ccccc1Cn2c(C)nc3ccccc23', 'O[C@@H](CNCCCOCCOCCc1cccc2ccccc12)c3ccc(O)c4NC(=O)Sc34', 'CC(C)c1ccc2Oc3nc(N)c(cc3C(=O)c2c1)C(=O)O', 'COc1ccc(cn1)C2=Cc3c(C)nc(N)nc3N([C@@H]4CC[C@H](CC4)OCCO)C2=O', 'NC(=NC#N)c1sc(Nc2ccccc2)nc1N', 'OB1N(C(=O)Nc2ccccc12)c3ccccc3', 'COc1ccc2ncc(C#N)c(CCN3CCC(CC3)NCc4cc5SCOc5cn4)c2c1', 'Cn1cncc1c2c3C(=O)N(CC4CC4)C(=O)N(CC5CC5)c3nn2Cc6ccnc7ccc(Cl)cc67', 'CN1CCN(CC1)c2ccc3N=CN(C(=O)c3c2)c4cc(NC(=O)c5cscn5)ccc4C', 'CS(=O)(=O)C1(CC1)c2cc(nc(n2)c3cccc4[nH]ccc34)N5CC6CCC(C5)O6', 'CNC1=Nc2ncccc2C(=NC1c3cccs3)c4occn4', 'C(CCCCNc1cc(nc2ccccc12)c3ccccc3)CCCNc4cc(nc5ccccc45)c6ccccc6', 'Clc1ccc(N2CCN(CC2)C(=O)CCCc3ccncc3)c(Cl)c1', 'COc1cc(ccc1N2CC[C@@H](O)C2)N3N=Nc4cc(sc4C3=O)c5ccc(Cl)cc5', 'CCC(COC(=O)c1cc(OC)c(OC)c(OC)c1)(N(C)C)c2ccccc2', 'CCCSc1ncccc1C(=O)N2CCCC2c3ccncc3', 'Oc1ncnc2scc(c3ccsc3)c12', 'OC1(CN2CCC1CC2)C#Cc3ccc(cc3)c4ccccc4']

            test_smiles_list_from_UniMol = ['c1ccc(COc2ccc(Nc3ncnc4[nH]nc(OCCN5CCCC5)c34)cc2)nc1', 'c1ccc(-c2cnc[nH]2)cc1', 'c1ccc(Nc2ccnc(Nc3cc(N4CCCC4)cc(N4CCOCC4)c3)n2)cc1', 'O=S1(=O)CCC(N2CC(Oc3nc4c(N5CCOCC5)nc(-n5cnc6ccccc65)nc4[nH]3)C2)CC1', 'c1cncc(CNc2ccc3nncn3n2)c1', 'O=C(CSc1nncn1-c1ccccc1)Nc1ccccc1', 'c1ccc(Cc2cnc3nc(-c4cccnc4)nn3c2)cc1', 'O=C(Nc1ccccc1)N1C2CCCC1CCC2', 'O=C1CC(=O)N(c2ccccc2)N1c1ccccc1', 'c1cnc(CNc2ccc3ncn(-c4cc(C5CC5)[nH]n4)c3n2)nc1', 'O=C(Nc1ccccc1)Nc1ncnc2cc(OCC3CCNCC3)ccc12', 'O=C1C[C@H]2CC(SC3CN(C4=NCCS4)C3)=CN12', 'O=C(CSc1ccncc1)Nc1ccc(C(=O)c2ccccc2)cc1', 'c1ccc(Nc2nccc(NC3CC3)n2)cc1', 'O=C1CC[C@@H]2C1CC[C@@H]1c3ccccc3CC[C@H]12', 'c1ccc2c(c1)CC[C@@H]1[C@@H]2CCC2CCC[C@H]21', 'O=C(NC1CCCCC1)c1cccnc1Sc1ccccc1', 'O=c1ccn(C2CC2)c2cc(N3CCOCC3)ccc12', 'O=C(Nc1ccccc1-c1ccccc1)OC1CCN(CCCCCCCCCNCCc2cccc3[nH]c(=O)ccc23)CC1', 'O=C(NS(=O)(=O)c1ccccc1)c1ccn(-c2ccccc2)c1', 'c1ccc2nc(O[C@@H]3CCN(c4ncnc5ccccc45)C3)cnc2c1', 'O=S(=O)(c1ccccc1)N1CCCC1', 'O=C(c1scc2c1OCCO2)N1CCCCC1', 'c1ccc2c(c1)ncn2-c1nc(N2CCOCC2)nc(N2CCOCC2)n1', 'O=C(CC1CCCCC1)N[C@H]1CCCN(C(=O)Cc2ccccc2)C1', 'c1ccc(C2CCN(Cc3c[nH]c4ccccc34)CC2)cc1', 'O=C(NC1CC1)c1cc2cccc(-c3ccnnc3)c2nn1', 'c1ccc(CNc2ccc3ncncc3c2)cc1', 'c1ncc(-c2nc(N3CCOCC3)nc3c2CCN3)cn1', 'c1ncc2c(n1)OCC(c1ccc(C3CCCCC3)cc1)=N2', 'c1cc(-c2nc(N3CCOCC3)c3sc(CN4CCNCC4)cc3n2)c2cn[nH]c2c1', 'c1nc(NC2CCOC2)cc(N2CCOCC2)n1', 'c1ccc(Nc2ccnc3cc(N4CCNCC4)ccc23)cc1', 'O=C1NC(=O)C(Cc2ccc(OCCc3ccccn3)cc2)S1', 'c1ccc(CCNc2ccc3nnc(-c4ccccc4)n3n2)cc1', 'C(#Cc1ccc2c(n1)OCCN2)c1ccccc1', 'O=C(c1ccccc1)N1CCN(S(=O)(=O)c2ccc3ccccc3c2)CC1', 'O=C(Nc1ccncc1)c1cccc(OCC2CC2)c1', 'O=S(=O)(c1ccccc1-c1ccccc1)N1CCCC1', 'O=S1(=O)c2ccccc2-c2ccccc21', 'c1ncc2ccsc2n1', 'c1ccc(-c2nn3c(-c4[nH]nc5c4CCC5)nnc3s2)cc1', 'O=c1[nH]c(=O)n(CC2CC2)c2nn(Cc3ccnc4ccccc34)c(-c3ncn[nH]3)c12', 'O=c1nc(NCc2ccccc2)s[nH]1', 'O=C(Cc1ccc(-n2cnnn2)cc1)NC1CCN(Cc2nc3c(s2)CCCC3)CC1', 'O=C(NCCCNC1CCN(Cc2ccccc2)CC1)c1ccccc1', 'c1ccc2c(c1)cnn2-c1cc(NC2CC2)n2nccc2n1', 'C(=NN=Cc1ccccc1)c1ccccc1', 'O=C(c1ccccc1)C1CCN(C(=O)c2ccco2)CC1', 'O=C(Nc1ccccc1)c1ccc(-c2ccccc2)cc1', 'c1ccc(-c2ncccn2)cc1', 'c1ccc2c(c1)ccn2-c1ccnc2ccccc12', 'O=C(CC1CCCCC1)N[C@H]1CCN(C(=O)Cc2ccccc2)C1', 'O=C(Cc1cc(Nc2ncnc3cc(OCCCN4CCCCC4)ccc23)n[nH]1)Nc1ccccc1', 'O=C(Nc1cnc(-c2ccccc2)[nH]c1=O)OCc1ccncc1', 'O=C(Nc1ccccc1)c1cc2ccccc2o1', 'O=C(Nc1ccccc1)c1ccc(N2CCOCC2)cc1', 'O=C(NCC12CC3CC(CC(C3)C1)C2)c1ccnc(N2CCCCC2)c1', 'C1=C/C(=C/c2ccc[nH]2)N=C1c1cc2ccccc2[nH]1', 'c1ccc(-c2c(-c3ccc(C4CCC4)cc3)ncc3ncccc23)cc1', 'O=C(Nc1cnc(C2CCCCC2)[nH]c1=O)OCc1ccccc1', 'c1ccc(-c2ncn(C[C@H]3CCCO3)c2-c2nc3cncnc3s2)cc1', 'O=C(Nc1ccc2ccn(Cc3ccc(-c4nn[nH]n4)cc3)c2c1)OC1CCCC1', 'c1ccc(C23CCC(CC2)CC3)cc1', 'c1ccc([C@@H]2CCCc3ccccc32)cc1', 'c1ccc(Nc2ccnc(Nc3cc(N4CCNCC4)cc(N4CCOCC4)c3)n2)cc1', 'C(CCN1CCNCC1)=C1c2ccccc2Sc2ccccc21', 'c1ccc(Nc2ncn[nH]2)cc1', 'O=c1[nH]c2cncnc2n1Cc1cccnc1', 'O=C(NC1CCCS(=O)(=O)C1)c1ccc(Oc2ccc(C#CC3CN4CCC3CC4)cc2)cc1', 'c1ccc(Nc2nc(Nc3ccc4c(c3)NCC4)nc3[nH]ccc23)cc1', 'O=C(NC1CCCCC1)c1ccc(NCC2CCCCC2)nc1', 'O=C(Cc1cnc(Nc2ncnc3cc(OCCCN4CCCCC4)ccc23)s1)Nc1ccccc1', 'O=C(NCC12CC3CC(CC(C3)C1)C2)c1cccc(-c2ccccc2)c1', 'O=C(CCCCC(=O)Nc1ccc2ncccc2c1)Nc1ccc2ncccc2c1', 'c1csc(-c2ccn3nccc3n2)c1', 'c1ccc(-c2ccc3ccccc3c2)cc1', 'O=C(Nc1cccc2cccnc12)c1ccc(N2C(=O)[C@H]3[C@H]4C=C[C@H](C4)[C@H]3C2=O)cc1', 'c1cc(-c2cnc3cc(N4CCCC4)ccn23)nc(Nc2ccc(N3CCNCC3)cc2)n1', 'O=c1ccc([C@]2(c3cccc(-c4ccccn4)c3)N=Cc3ccccc32)c[nH]1', 'C1=C2CCC(CCc3ccccc3)C2Cc2cnn(-c3ccccc3)c21', 'c1ccc(-c2csc(-c3ccccc3)n2)cc1', 'c1cnc2ncnn2c1', 'O=C(COCCNS(=O)(=O)c1ccccc1)N1CCN(C2CCNCC2)CC1', 'S=c1[nH]cc(Cc2ccccc2)[nH]1', 'O=C(NC1CCCC1)C(c1ccccc1)N(C(=O)c1ccco1)c1ccccc1', 'c1ccc(Nc2ccnc(Nc3cc(CN4CCOCC4)cc(N4CCOCC4)c3)n2)cc1', 'O=C(CCN1CCC(c2ccc(C(=O)Nc3ccccc3)cc2)CC1)Nc1ccccc1', 'O=C1NC(=O)/C(=C/c2ccccc2N2CCNCC2)S1', 'O=C1C=C(NCc2ccc(Nc3ncc4cc(-c5ccncc5)ccc4n3)cc2)CCN1', 'c1ccc(-n2cc(C3CCNCC3)c3ccccc32)cc1', 'O=C(c1ccccc1)c1ccccc1NCCc1ccc(OCCc2csc(N3CCNCC3)n2)cc1', 'c1cnnc(N2CC(COc3ccc(-c4ncon4)cc3)C2)c1', 'O=C(Nc1nnn[nH]1)c1cc(Oc2ccccc2)c2ccccn2c1=O', 'c1ccc2c(c1)ccn2-c1cc(Nc2cc[nH]n2)n2nccc2n1', 'O=C(NCC1CCCCN1)c1ccccc1', 'O=C1C[C@H]2CC[C@]1(CS(=O)(=O)NC1CCN(c3ccccn3)CC1)C2', 'O=P(Nc1cccnc1)(c1ccccc1)c1ccccc1', 'c1ccc(CNc2ccccc2)cc1', 'O=c1[nH]c2cccc(CCNCCc3cccc(CN4CCC(c5ccccc5)CC4)c3)c2s1', 'c1ccc(C(O[C@@H]2C[C@@H]3CC[C@H](C2)N3)c2ccccc2)cc1', 'O=C(NS(=O)(=O)c1ccc(N[C@H](CCN2CCOCC2)CSc2ccccc2)cc1)c1ccc(N2CCN(CC3=C(c4ccccc4)CCCC3)CC2)cc1', 'O=c1ccccc2ccccc12', 'O=C(NS(=O)(=O)c1ccc(Oc2ccccc2)cc1)c1cccc(-c2ccccc2)c1', 'c1ccc(SCCCOc2ccc3cnoc3c2)cc1', 'O=C(CC1CCCCC1)N[C@H]1C[C@@H]2CC[C@H]1N(CCc1ccccc1)C2', 'O=C(NCc1ccccc1)C1c2ccccc2C(=O)N1CC1CCC1', 'c1ccc(OCc2ccc3ccccc3n2)cc1', 'c1ccc(-c2cnc3ccccc3n2)cc1', 'c1ccc(Nc2ccnc(Nc3cc(N4CCCCC4)cc(N4CCOCC4)c3)n2)cc1', 'c1ccc2nc(-c3ccncc3)ccc2c1', 'c1cncc(C2CCCN2)c1', 'O=C(NC1CC1)[C@@H]1CCCC[C@H]1C(=O)N1CCN(c2nc3ncccc3s2)CC1', 'O=C(NCC12CC3CC(CC(C3)C1)C2)c1cccc(CN2CCNCC2)c1', 'O=C1CSC(=O)N1C(CCc1ccccc1)COc1ccc(-c2ccccc2)cc1', 'O=C(Nc1ccc(C2CCOCC2)nc1)c1cc(=O)c2cccc(-c3cn[nH]c3)c2o1', 'O=C(C1CCCNC1)N1CCC(Cc2ccccc2)CC1', 'O=C(c1ccccc1)N1CCC(n2c(=O)[nH]c3ccccc32)CC1', 'N=c1[nH]c2ccccc2n1CC(=O)c1ccccc1', 'O=C(NC1CCCC1)C(c1ccncc1)N(Cc1ccco1)C(=O)c1ccc(-c2ccccc2)[nH]1', 'O=C(c1ccc(C(=C2CCN(Cc3cscn3)CC2)c2cccc3cccnc23)cc1)N1CCCC1', 'O=C(Nc1ccc(N2CCOCC2)cc1)c1cc(=O)c2cccc(N3CCCC3)c2[nH]1', 'c1ccc(Nc2nc(SCc3cccs3)n[nH]2)cc1', 'O=C(NCC1CN(c2ccnc3ccccc23)C(=O)O1)C1CC1', 'O=S(=O)(CC1CCCO1)c1ccc(Nc2nccc(-c3cnc[nH]3)n2)cc1', 'c1ccc(-c2cc(N(CCCN3CCCCCC3)Cc3ccc4c(c3)OCO4)on2)cc1', 'c1ccc2c(c1)CCN2c1cc(NC2CC2)n2nccc2n1', 'O=C(NC1CC1)[C@@H]1CCCC[C@H]1C(=O)N1CCN(c2ccc3cnncc3c2)CC1', 'c1csc(-c2ccc(CNC3CCCC3)cc2)c1', 'c1ccc2c(c1)ccn2-c1cc(NC2COC2)n2nccc2n1', 'O=C(/C=C/CN1CCCCC1)Nc1ccc2ncnc(Nc3ccccc3)c2c1', 'N=C1CCCCOC(=O)C[C@@H](O[C@H]2CCCCO2)C[C@@H](O[C@H]2CCCCO2)CCC1', 'O=c1[nH]c2cccc(CCNCCc3cccc(CN4CCCC4)c3)c2s1', 'C1=NC2(N=C1)c1cc(-c3cccnc3)ccc1CC21CCCCC1', 'O=C(NCc1ccccc1)Nc1nccs1', 'O=c1c2ccccc2nc(-c2ccccc2)n1-c1ccc(C2CCC2)cc1', 'O=C(NC1CC1)[C@@H]1CCCC[C@H]1C(=O)N1CCc2oc3ccccc3c2C1', 'c1ccc(C[n+]2cc3ccccc3c3nn(-c4ccccc4)cc32)cc1', 'O=C1c2nc3ccnn3cc2CN1Cc1ccccc1', 'O=c1cc(N2CCOCC2)oc2c(-c3cccc(-c4ccsc4)c3)cccc12', 'O=C(NC1CC1)[C@@H]1CCCC[C@H]1C(=O)N1CCN(c2nc3ccccc3s2)CC1', 'O=C(Nc1cnc(-c2cccs2)[nH]c1=O)OCc1ccccc1', 'c1csc(-c2ccc(OCCN3CCCCC3)cc2)c1', 'O=C(c1scc2c1CCCC2)N1CCCCC1', 'O=c1cnc2cccnc2n1CCN1CCC(NCc2cc3c(cn2)OCCO3)CC1', 'O=C1[C@H]2[C@H]3CC[C@H](C3)[C@H]2C(=O)N1[C@H]1CC[C@H](C(=O)Nc2cccnc2)CC1', 'O=S(=O)(Nc1nc(-c2ccccc2)nn1Cc1ccccc1)c1ccccc1', 'O=c1n(CCCN2CCN(c3ccccc3)CC2)nc2ccccn12', 'O=C(CN1CCOCC1)Nc1ccc(-c2ccccc2)[nH]c1=O', 'c1nc2c(cc1COC1CCCCC1)OCCN2', 'O=C(CNCc1ccc(-c2ccccc2)cc1)NC1CC1', 'O=C(NCCCN1CCC(Oc2ccccc2)CC1)n1c(=O)[nH]c2ccccc21', 'C(#Cc1ccc(CN2CCOCC2)cc1)c1ccccc1', 'c1ccc(Sc2c[nH]c3ccncc23)cc1', 'O=C(CC1CCCCC1)N[C@H]1CCN(CCc2ccccc2)C1', 'O=C(c1ccccc1)C1CCN(C(=O)C2CC2)CC1', 'c1ccc(Cn2nnc(-c3cccc(OCc4ccc5ccccc5n4)c3)n2)cc1', 'O=C(c1c(Cc2c[nH]c3ncccc23)sc2[nH]c(=O)[nH]c(=O)c12)N1CCCO1', 'O=C(NC1CCCCC1)NS(=O)(=O)c1ccc(CCNC(=O)c2cnccn2)cc1', 'c1cnc(CNc2ncc3ncn(-c4cc[nH]n4)c3n2)nc1', 'C1=NC(c2cccc(-c3cncnc3)c2)CCS1', 'O=C(Nc1ccccc1)N1CCN(CC2CCCNC2)CC1', 'O=C(NCc1ccccc1)c1cnn(-c2ccccc2)c1NS(=O)(=O)c1ccccc1', 'c1ccc(-c2nnc(-c3ccccc3)s2)cc1', 'O=C(c1cccc(Cc2n[nH]c(=O)c3ccccc23)c1)N1CCN(c2ncccn2)CC1', 'c1csc(COc2ncc3ncn([C@H]4CCCO4)c3n2)n1', 'c1nc2c(s1)CCCC2', 'O=C(NCC12CC3CC(CC(C3)C1)C2)c1cccc(O[C@H]2CCCNC2)c1', 'c1cc2c3c(c1)O[C@H]1c4[nH]c5ccccc5c4CC4[C@@H](C2)N(CC2CC2)CC[C@@]341', 'O=C(NCCc1ccccc1)c1cccc2cc(-c3ccc4c(C(=O)NCCc5ccccc5)cccc4c3)ccc12', 'O=C(Nc1ccc(S(=O)(=O)N2CCOCC2)cc1)c1ccccc1NS(=O)(=O)c1cccs1', 'O=C(NCc1cccnc1)c1cc(C2CC2)nc2[nH]ncc12', 'O=C1CN(Cc2ccnc(N3CCOCC3)c2)C(=O)N1c1ccccc1', 'c1ccc(C2CCCN(CC3CC3)C2)cc1', 'c1cncc(-c2nc3ccccc3[nH]2)c1', 'O=C(Nc1ccccn1)c1cccc(OCCc2cccnc2)c1', 'O=C(Nc1cccc(C(=O)Nc2ccon2)c1)c1cnn(-c2ccccc2)c1', 'O=C(CSc1nnc(-c2ccccc2)o1)N1CCc2ccccc21', 'c1cc(COc2cnc(N3CCN(c4noc(C5CC5)n4)CC3)nc2)ccn1', 'c1ccc2cc(Nc3ncnc4ccccc34)ncc2c1', 'c1cc(N2CCOCC2)nc(-c2cnc3[nH]ccc3c2)n1', 'c1ccc(CNCCNCc2ccccc2)cc1', 'O=S1(=O)NC(Cc2ccccc2)Nc2ccccc21', 'O=C(Nc1cccc(Nc2ncnc3ccc(OCCN4CCCC4)cc23)c1)c1ccnc(N2CCOCC2)c1', 'O=C1CCOc2nc(/C=C/c3ccccc3)ccc21', 'c1cc(-c2cc(-c3ncc(CN4CCNCC4)o3)c3cn[nH]c3c2)c2cc[nH]c2c1', 'O=C1CN(C(=O)c2ccccc2)c2ccccc2N1', 'O=C(NCCCN1CCC(Oc2ccccc2)CC1)c1n[nH]c2ccccc12', 'O=C(NC(c1ccccc1)C1CC1)C1CCN(c2ncnc3[nH]ccc23)CC1', 'O=c1[nH]c(=O)c2cc(Cn3ccnc3)sc2[nH]1', 'O=C(NCCCSC[C@@H]1CC[C@H](n2cnc3cncnc32)O1)NCc1ccccc1', 'O=C(Nc1ccc(N2CCNCC2)nc1)c1nnc(Nc2ccccc2)o1', 'O=C(Nc1cccnc1)c1cncc(-c2ccc(S(=O)(=O)N3CCOCC3)cc2)n1', 'O=C1Nc2ccccc2CCC1NCCCc1ccccc1', 'O=C(NCc1nc2ncccc2[nH]1)c1cc2ccccc2o1', 'c1ccc(-c2cncnn2)cc1', 'O=C(Nc1ccc(N2CCOCC2)cc1)c1nnc(NC(=O)C2(c3ccccc3)CCCC2)o1', 'c1ccc(CN2CCN(C3=Nn4cnnc4CC3)CC2)cc1', 'O=c1c(NCc2ccco2)c(Nc2ccccc2)c1=O', 'O=C(NCc1cn(-c2ccnc3ccccc23)nn1)c1ccccc1', 'c1ccc(-c2cnc3ncncc3n2)cc1', 'c1ccc(-n2cncn2)cc1', 'c1ccc(-c2cc3ccccc3cn2)nc1', 'c1ccc(COc2csc(-n3cnc4ccccc43)c2)cc1', 'O=C1CCN(C(=O)c2ccccc2)c2ccccc2N1', 'O=C(Nc1cccnc1)C1CCCN1c1nc(Nc2cc(C3CC3)[nH]n2)c2cccn2n1', 'O=C(NCC(=O)N1CCOCC1)Nc1nc2ccc(-c3cccnc3)cn2n1', 'c1ccc(-c2nnc[nH]2)cc1', 'c1ccc2c(c1)Cc1cccnc1N1CCNCC21', 'O=C1CCOc2nc(C#CC3CCCC3)ccc21', 'O=S(=O)(N[C@@H]1CCN(Cc2ccc(-c3ccccc3)cc2)C1)c1ccccc1', 'O=C1Cc2ccccc2Nc2ccccc21', 'c1ccc(OC2CCN(CC3CCN(c4ccccn4)CC3)CC2)cc1', 'O=C1CCNc2ccccc2N1', 'O=C(Cn1ccccc1=O)c1ccn(Cc2ccccc2)c1', 'c1ccc(-c2cc(Nc3cc[nH]n3)n3nccc3n2)cc1', 'O=C1COC2(CCN(S(=O)(=O)c3ccc(-c4ccc5cnccc5c4)cc3)CC2)CN1C1CC1', 'c1nc(-c2cnn(CC3CCCC3)c2)c2cc[nH]c2n1', 'O=C(Nc1ccc(-c2cccnc2)[nH]c1=O)OCc1ccccc1', 'O=C(Nc1nc2ccccc2n1CCN1CCOCC1)c1ccccc1', 'O=C(NCc1nc2ccccc2[nH]1)c1cccs1', 'c1ccc(-c2cc3cccn3nc2-c2ccc(C3CCC3)cc2)cc1', 'O=c1[nH]c2ccccc2o1', 'O=c1[nH]cnc2scc(-c3cccs3)c12', 'O=C1CN2Cc3ccccc3N=C2N1', 'O=C1CCOc2nc(CCc3ccccc3)ccc21', 'c1ccc(-c2cncc(-c3cccnc3)c2)cc1', 'c1cnc2c(CCN3CCC(NCc4cc5c(cn4)OCCO5)CC3)ccnc2c1', 'O=c1ncn2nc(Sc3ccccc3)ccc2c1-c1ccccc1', 'O=S(=O)(Nc1ccnn1C1CCOC1)c1ccccc1', 'O=c1[nH]cnc2[nH]ncc12', 'O=c1c2ccccc2nc(Cn2nc(-c3ccccc3)c3cncnc32)n1-c1ccccc1', 'O=C(C1CCCCC1)N1CC(=O)N2CCc3ccccc3C2C1', 'O=C(Nc1ccccc1)c1cnns1', 'O=C(COc1ccccc1)N1CCOCC1', 'O=C(NCC12CC3CC(CC(C3)C1)C2)c1cccc(CN2C[C@@H]3C[C@H]2CN3)c1', 'c1cnc2sc3c(N4CCNCC4)ncnc3c2c1', 'c1ccc(CCN2CCC(Cc3ccccc3)CC2)cc1', 'O=C(Nc1nc2c(=O)[nH]cnc2s1)c1ccccc1', 'C1CCC(C(CC2CCCCN2)C2CCCCC2)CC1', 'C(=C\\c1cncn1Cc1ccccc1)\\Cc1cccs1', 'O=c1[nH]n(C(c2ccncc2)C2CC2)c(=O)c2cc3ccccc3nc12', 'O=C(N[C@H]1CC[C@@H](Nc2ncc3ccccc3n2)CC1)c1ccccc1', 'O=C(NCc1ccccc1)[C@@H]1CCCC[C@H]1C(=O)N1CCN(Cc2ccccc2)CC1', 'O=C(Nc1ccccc1)c1ccc(C2CCN(Cc3ccc(C(=O)NC4CC4)cc3)CC2)cc1', 'c1ccc(-c2ccc(COc3ccc4c(c3)CCC(CCN3CCCCC3)C4)cc2)cc1', 'O=c1nc[nH]c2[nH]cnc12', 'O=C(c1c(Cc2c[nH]c3ccccc23)sc2[nH]c(=O)[nH]c(=O)c12)N1CCCC1', 'O=C(NCCSCc1ccccc1)c1ccccc1', 'O=C(CC1CCCCC1)N[C@H]1CCN(C(=O)c2ccccc2)C1', 'O=C1CCOC(=O)CNC(=O)CNC(=O)CNC(=O)CNC(=O)CN1', 'O=C(Nc1cccc(Nc2ccnc(NC3CCNCC3)n2)c1)c1ccnc(N2CCOCC2)c1', 'c1ccc(Sc2c[nH]c3cccnc23)cc1', 'N=C(NCc1ccccc1)SCCCc1c[nH]cn1', 'O=C(NCC1CCCCCC1)c1cccc(-c2cccnc2N2CCCCC2)c1', 'O=C(c1c(Cc2cccc3ncccc23)sc2[nH]c(=O)[nH]c(=O)c12)N1CCCC1', 'O=C(NCc1ccc(Nc2ncc3cc(-c4ccncc4)ccc3n2)cc1)[C@H]1CCCCN1', 'O=C(CNC(=O)c1ccon1)C[C@@H](Cc1ccccc1)C(=O)NCC[C@@H]1CCNC1=O', 'O=c1[nH]n(Cc2ccccn2)c(=O)c2c(=O)c3ccccc3[nH]c12', 'c1ccc(C(c2ccccc2)N2CCCNCC2)cc1', 'c1ccc(-n2nnnc2NCc2ccccc2Oc2ccccn2)cc1', 'O=S(=O)(Cc1ccccc1)N1CCCN(Cc2ccccc2)CC1', 'c1ccc(CN2CCN(C(c3ccccc3)c3ccccc3)CC2)cc1', 'O=c1[nH]c2cccc(CCNCCc3ccc(-c4ccccc4CNCc4ccccc4)cc3)c2s1', 'c1ccc(-c2cn(CCCOc3cccc4ccccc34)c3ccccc23)cc1', 'O=C(CCN1CCCCC1)c1ccccc1', 'O=C(Nc1ccccc1)c1cccnc1Oc1ccccc1', 'O=C(NCC12CC3CC(CC(C3)C1)C2)c1cccc(CC2CCNCC2)c1', 'O=C(Nc1ccccc1)N1CCN(C(=O)[C@H]2CNCCN2)CC1', 'O=C(c1ccccc1)N1CCNCC1', 'c1ccc(-c2c(-c3ccc(C4CCC4)cc3)ncc3ccncc23)cc1', 'O=C(Cc1cc2c(CCN3CCNCC3)c(-c3ccccc3)[nH]c2s1)N1C2CCC1CC2', 'O=C(NCc1ccc(Nc2ncc3cc(-c4ccncc4)ccc3n2)cc1)C1CCCCN1', 'O=C(Cc1ccccc1)OC1CCCCC1', 'C1=C[C@@H](c2ccccc2)c2cnccc2N1', 'O=S(=O)(Nc1cncc(-c2cncs2)c1)c1cncs1', 'c1ccc(Nc2ncnc3cc(OCCn4cncn4)ccc23)cc1', 'O=c1[nH]c(=O)c2cn(Cc3ccccc3)cc2[nH]1', 'O=C(COc1cccc2ncnc(Nc3ccc4c(cnn4Cc4ccccn4)c3)c12)N1CCOCC1', 'O=C(NCc1ccnc2ccccc12)c1ccc(Oc2ccc(C#CC3CN4CCC3CC4)cc2)cc1', 'O=S(=O)(c1cccnc1)N1CCCC1', 'c1cc2nncn2nc1N1CCCCC1', 'O=C(Oc1ccccc1)N1CCOCC1', 'O=C(NCC1CCCCC1)c1cccc(-n2ncc(=O)[nH]c2=O)c1', 'O=C(c1c(Cn2ccnc2)sc2[nH]c(=O)[nH]c(=O)c12)N1CCCC1', 'O=c1ccn2c3c(cccc13)CCC2', 'O=C(Cn1cc(Oc2ncnc3ccccc23)cn1)Nc1ccccc1', 'c1ccc(Oc2cccc3cccnc23)cc1', 'c1ccc(-c2cnn[nH]2)cc1', 'O=C(CC12CC3CC(CC(C3)C1)C2)Nc1cccc2c(=O)[nH]ccc12', 'O=C(Nc1cccc(Nc2ncnc3ccc(OCCN4CCOCC4)cc23)c1)c1ccnc(N2CCOCC2)c1', 'O=c1[nH]c2cccc(CCNCCc3ccccc3CNCCc3ccccc3)c2s1', 'O=C(COCc1cscn1)N1CCN(c2ccccc2)CC1', 'c1ccc(-c2ncc3c(n2)CCSC3)cc1', 'O=c1cc(N2CCOCC2)oc2ccccc12', 'O=C1/C=C\\C=C\\CCCCCCCC/C=C/OC2Oc3ccc4c(c3C2=O)C(=O)C=C(N1)C4=O', 'O=C(Cc1ccccc1)Nc1nnc(CCSCCc2nncs2)s1', 'c1ccc(N2CCOc3ccccc32)[nH+]c1', 'c1ccc([SH+]c2ccc3nnnn3n2)cc1', 'O=c1ccoc2cc(OCc3ccccc3)ccc12', 'c1ccc(-c2nnc3ccc(OCc4ncn[nH]4)nn23)cc1', 'O=C1C=C(NCc2ccc(Nc3ncc4cc(-c5ccncc5)ccc4n3)cc2)CC1', 'c1ccc2c(c1)N=C(N1CCNCC1)c1ccccc1N2', 'c1nc(NC2CC3CCC2C3)cc(N2CCOCC2)n1', 'c1ccc(CNc2cnccn2)nc1', 'O=C(NC1CCN(CCC(c2ccccc2)c2ccccc2)CC1)C1CCC1', 'c1ccc(CC(Cc2nc3ccccc3[nH]2)c2nc3ccccc3[nH]2)cc1', 'O=c1[nH]nc2c3cc(-c4ccccc4)c(-c4ccc(C5CCC5)cc4)nc3ccn12', 'O=S(=O)(Cc1ccccc1)c1nnco1', 'c1ccc(N2CCc3cc4ccccc4nc32)cc1', 'O=C(Oc1ccccc1)c1cccc2ccccc12', 'c1ccc(CN2CCC(NCCCOc3ccccc3)CC2)cc1', 'O=c1cc(CCc2cccc(-c3ccccc3)c2)nc[nH]1', 'c1ccc(N2CC3COc4ccccc4C3=N2)cc1', 'c1ccc(Cc2cn[nH]c2)cc1', 'O=C1N(c2ccccc2)Cc2cncnc2N1Cc1ccccc1', 'O=C(NCCCNC[C@@H]1CC[C@H](n2cnc3cncnc32)O1)Nc1ccccc1', 'O=C(Nc1ccccc1)Nc1cc(Nc2ccc(N3CCNCC3)cc2)ncn1', 'c1ccc2c3c([nH]c2c1)[C@@H]1C[C@@H]2CCCC[C@H]2CN1CC3', 'c1cc(CCCOc2ccc(C3=NCCO3)cc2)on1', 'c1ccc2c(c1)C1CCC2c2ccccc21', 'O=c1s[nH]c2ccc(S(=O)(=O)Nc3ccccc3)cc12', 'O=C(NC1C2CC3CC(C2)CC1C3)c1cnc(N2CCOCC2)nc1C1CC1', 'O=C1C=C2CC[C@@H]3[C@H](C(=O)CC4CCC[C@H]43)C2CC1', 'c1cnc(CNc2nccc(Nc3cc(C4CC4)[nH]n3)n2)nc1', 'O=S(=O)(CC1CCOCC1)N1CCC(CCc2ccncc2)CC1', 'O=C(NCC12CC3CC(CC(C3)C1)C2)c1cccc(CN2CC3CNCC(C2)O3)c1', 'O=C1CCOc2cc(COc3ccccc3)ccc21', 'c1ccc(Nc2ncnc3ccc(OCCCN4CCCC4)cc23)cc1', 'C1=NC2(N=C1)c1ccccc1CC21CCCCC1', 'O=C(COc1ccccc1)c1ccccc1', 'O=C(CCS(=O)(=O)c1cccs1)Nc1ccccc1', 'O=C(OC1CN2CCC1CC2)C(c1ccccc1)c1ccccc1', 'O=c1cc(-c2ccccc2)nc(SCCc2ccccc2)[nH]1', 'O=C(c1ccc(-c2ccccc2)o1)N(Cc1ccccn1)c1ccc(N2CCNCC2)cc1', 'c1cncc(C2CCN(CC34CC(c5ccccc53)c3ccccc34)CC2)c1', 'c1ccc(Nc2cc(Nc3ccccc3)[nH]n2)cc1', 'O=S(=O)(Nc1ccnn1C1CCCC1)c1ccccc1', 'c1ccc(Cn2cnc3ccccc32)cc1', 'O=c1[nH]c2cccc(CCNCCCOCCOCCc3cccc4ccccc34)c2s1', 'O=c1c2ccccc2oc2ncccc12', 'O=c1c(-c2cccnc2)cc2cncnc2n1C1CCCCC1', 'c1ccc(Nc2nccs2)cc1', 'O=C1Nc2ccccc2BN1c1ccccc1', 'c1ccc2c(CCN3CCC(NCc4cc5c(cn4)OCS5)CC3)ccnc2c1', 'O=c1c2c(-c3cnc[nH]3)n(Cc3ccnc4ccccc34)nc2n(CC2CC2)c(=O)n1CC1CC1', 'N=C(Nc1ccccc1)Nc1cccc2ccccc12', 'O=C(Nc1cccc(-n2cnc3ccc(N4CCNCC4)cc3c2=O)c1)c1cscn1', 'c1cc(-c2nc(C3CC3)cc(N3CC4CCC(C3)O4)n2)c2cc[nH]c2c1', 'C1=Nc2ncccc2C(c2ncco2)=NC1c1cccs1', 'c1ccc(-c2cc(NCCCCCCCCNc3cc(-c4ccccc4)nc4ccccc34)c3ccccc3n2)cc1', 'O=C(CCCc1ccncc1)N1CCN(c2ccccc2)CC1', 'O=c1c2sc(-c3ccccc3)cc2nnn1-c1ccc(N2CCCC2)cc1', 'O=C(OCCc1ccccc1)c1ccccc1', 'O=C(c1cccnc1)N1CCCC1c1ccncc1', 'c1ncc2c(-c3ccsc3)csc2n1', 'C(#CC1CN2CCC1CC2)c1ccc(-c2ccccc2)cc1', 'O=C(NCCc1ccccc1)c1ccccc1', 'O=C(CSc1ncnc2sccc12)N1CCN(C(=O)c2ccco2)CC1', 'O=c1ccn(Cc2ccccc2)c(=O)n1Cc1ccccc1', 'C1=Nc2ncccc2C(c2ccco2)=NC1c1cccs1', 'c1cc2c(cc1-c1cnc3n1CCC3)OCCO2', 'O=c1ccn(Cc2ccccc2)c2sc(-c3ccccc3)c(CNCc3ccccc3)c12', 'c1ccc(-c2c(-c3ccc(C4CCC4)cc3)ncc3cccnc23)cc1', 'O=C1c2cc[nH]c2CCC1CN1CCOCC1', 'O=C(NC1CCCCC1)NS(=O)(=O)c1ccc(CCNC(=O)N2CC=CC2=O)cc1', 'c1ccc(-c2ncc3nccn3c2-c2ccc(C3CCC3)cc2)cc1', 'c1ccc(-c2nnc3ccc(NC4CC4)nn23)cc1', 'O=C(CCC(=O)Nc1ccc2ncccc2c1)Nc1ccc2ncccc2c1', 'C(=C/c1ccccc1)\\c1ccccc1', 'O=C(Nc1cc(-c2ccncc2)c[nH]c1=O)[C@H](Cc1ccccc1)NC1(c2ccccn2)CC1', 'c1cc2c3c(c[nH]c3c1)C[C@H]1NCCC[C@H]21', 'O=C(c1c(Cc2ccnc3ccccc23)sc2[nH]c(=O)[nH]c(=O)c12)N1CCCO1', 'c1ccc(-c2ccn3ncnc3c2-c2ccc(C3CCC3)cc2)cc1', 'O=S(=O)(CCCOCCc1cc2ccccc2o1)CCNCCc1cccc2ncsc12', 'O=c1cccnn1-c1ncccn1', 'c1ccc(Nc2ccnc3ccc(NCCN4CCCCC4)cc23)cc1', 'C(=C/c1n[nH]c2cc(Sc3ccccc3)ccc12)\\c1ccccn1', 'c1ccc(Nc2ccn[nH]2)cc1', 'O=C(CSc1ccncc1)Nc1cccc(S(=O)(=O)N2CCCc3ccccc32)c1', 'O=C(NCc1cncc(-c2ccccc2)c1)c1ccccc1', 'O=C(N[C@H]1CN2CCC1CC2)c1ccc(-c2ccccn2)s1', 'O=C(Nc1ccc(S(=O)(=O)c2ccccc2)cc1)c1cccc(Cc2ccccc2)c1', 'O=C(Nc1ccccc1)c1ccncn1', 'c1ccc(COc2ccc(N3CCNCC3)cc2)cc1', 'c1ccc([C@H]2CN3CCSC3=N2)cc1', 'c1nc(NC2CCCC2)cc(N2CCOCC2)n1', 'c1ccc(Nc2nc(-c3ccccc3)cs2)cc1', 'O=C(c1c(Cc2ccccc2)sc2[nH]c(=O)[nH]c(=O)c12)N1CCCCC1', 'c1ccc2c3c([nH]c2c1)C[C@@H]1CCN(CCN2CCOCC2)C[C@H]1C3', 'O=C(c1ccc2[nH]ccc2c1)N1CCN(Cc2ccccc2)CC1', 'c1ccc(-c2nnc(N3CCNCC3)c3ccccc23)cc1', 'O=C(O[C@H]1CN2CCC1CC2)N(Cc1ccccc1)c1ccccc1', 'c1cc(-c2cc3c(N[C@H]4CCCNC4)nccc3s2)ccn1', 'O=c1[nH]c(-c2ccncc2)cc2onc(-c3ccccc3)c12', 'O=C(NC1CC1)c1cc2cccnc2s1', 'c1ccc(Nc2nc(NCC3CCCO3)c3ccccc3n2)cc1', 'c1ccc(CN[C@H]2CCC[C@H](Nc3ccc4ccccc4n3)C2)cc1', 'C(#Cc1[nH]cnc1-c1ccccc1)c1ccncn1', 'O=C(NC1C2CC3CC(C2)CC1C3)c1cnn(-c2ccccc2)c1C1CC1', 'c1ccc2c(c1)nc1n2CCCCC1', 'O=c1ccn(-c2ccccc2)c2nc(N3CC4CC4C3)ccc12', 'O=C(NCCCN1CCC(Oc2ccccc2)CC1)c1c[nH]c(=O)c2cc(S(=O)(=O)NC3CC3)ccc12', 'O=C(Nc1ccc2cnn(Cc3ccc(C(=O)NS(=O)(=O)c4ccccc4)cc3)c2c1)OC1CCCC1', 'O=C(Cc1ccccc1)N1CCN(c2ncnc3c2CCC3)CC1', 'O=C(Nc1cccc(COc2ccccc2)c1)c1ccccc1', 'O=C1NC(=O)C(=Cc2ccccc2)S1', 'O=C(c1ccccc1)C1CCN(CCn2c(=O)[nH]c3ccccc3c2=O)CC1', 'O=S(=O)(CCCOCCSc1ccccc1)CCNCCc1cccc2ncsc12', 'O=C1c2ccccc2NCN1c1ccccc1', 'O=c1oc2ccccc2cc1Cc1ccccc1', 'O=C1OCc2ccccc2N1C1CCN(c2ccccn2)CC1', 'C1=NC(c2ccco2)Nc2ccccc21', 'O=C(Nc1cccc(NC(=O)c2cccc(N3CCOCC3)c2)c1)c1ccc(OCc2ccccn2)cc1', 'O=c1c(NCc2ccccc2)c(Nc2ccccc2)c1=O', 'O=C1CCC[C@@H](CC[C@H]2CC=CC3=CCCC[C@@H]32)O1', 'O=c1[nH]c2cccc(CCNCCSCCCOCCc3ccccc3)c2s1', 'c1cc(-c2nccc(N3CCOCC3)n2)c2[nH]ccc2c1', 'c1cnc(N2CCN(c3ncc(OCc4ccncc4)cn3)CC2)nc1', 'O=C1OCCN1c1ccc(N2CCOCC2)cc1']
    
    elif args.dataset == "qm9":
        full_dataset = QM9Dataset(return_types=args.required_data,  target_tasks=args.targets, device=device,
                          dist_embedding=args.dist_embedding, num_radial=args.num_radial)

        test_smiles_list_from_GraphMVP = []
        # actually in 3DInfomax they only use random splits for qm9 for some reason

    split_indices = full_dataset.get_idx_split()
    if args.force_random_split == True:
        all_idx = get_random_indices(len(full_dataset), args.seed_data)
        split_indices["train"] = all_idx[:len(split_indices["train"])]
        split_indices["valid"] = all_idx[len(split_indices["train"]):len(split_indices["train"])+len(split_indices["valid"])]
        split_indices["test"] = all_idx[len(split_indices["train"])+len(split_indices["valid"]):]

    test_indices = split_indices["test"]

    test_smiles_list = df_smiles[test_indices].tolist()

    common_smiles_GraphMVP = [smiles for smiles in test_smiles_list if smiles in test_smiles_list_from_GraphMVP]
    common_smiles_UniMol = [smiles for smiles in test_smiles_list if smiles in test_smiles_list_from_GraphMVP]

    print(f"Task name: {args.dataset}")
    print(f"#molecules in test set: {len(test_smiles_list)}")

    # Individual checks
    print(f"#common SMILES with GraphMVP: {len(common_smiles_GraphMVP)}")
    print(f"#common SMILES with UniMol: {len(common_smiles_UniMol)}")
    print(f"#test SMILES: {len(test_smiles_list)}")

    print(f"GraphMVP matches test set: {len(common_smiles_GraphMVP) == len(test_smiles_list)}")
    print(f"UniMol matches test set: {len(common_smiles_UniMol) == len(test_smiles_list)}")


def test_custom_80_10_10_scaffold_split_matches_ground_truth():
    
    # for better console readability.
    import warnings
    warnings.filterwarnings(
        "ignore",
        message="You are using `torch.load` with `weights_only=False`",
        category=FutureWarning
    )

    return_types = ["dgl_graph", "targets"]
    noise_level = 0.0
    device = torch.device("cuda:0")

    datasets = ["ogbg-molbace", "ogbg-molbbbp", "ogbg-molclintox", "ogbg-molmuv", "ogbg-molsider", "ogbg-moltox21", "ogbg-moltoxcast", "ogbg-molhiv", "ogbg-molesol", "ogbg-molfreesolv", "ogbg-mollipo"]
    #datasets = ['ogbg-moltoxcast', 'ogbg-molfreesolv']

    for dataset_name in tqdm(datasets, desc="Processing datasets"):    
        dataset = OGBGDatasetExtension(return_types=return_types, device=device, name=dataset_name, noise_level=noise_level)

        split_idx = dataset.get_idx_split() # ground truth

        all_idx = dataset.get_all_indices()

        custom_split = scaffold_split(dataset_name)
        train_idx = custom_split['train']; valid_idx = custom_split['valid']; test_idx = custom_split['test']

        print("Train, valid and test contains all indices: ", len(set(train_idx + valid_idx + test_idx)-set(all_idx))==0)

        train_is_equal = all(train_idx == split_idx['train']) 
        valid_is_equal = all(valid_idx == split_idx['valid']) 
        test_is_equal = all(test_idx == split_idx['test']) 
        print(f"{dataset_name} - Train idx match: {train_is_equal}")
        print(f"{dataset_name} - Valid idx match: {valid_is_equal}")
        print(f"{dataset_name} - Test idx match:  {test_is_equal}")
        
        if train_is_equal == False or test_is_equal == False or test_is_equal == False:
            print(f"Indices in train_idx but not in split_idx['train']: {set(train_idx) - set(split_idx['train'])}")
            print(f"Indices in split_idx['train'] but not in train_idx: {set(split_idx['train']) - set(train_idx)}")

            print(f"Indices in valid_idx but not in split_idx['valid']: {set(valid_idx) - set(split_idx['valid'])}")
            print(f"Indices in split_idx['valid'] but not in valid_idx: {set(split_idx['valid']) - set(valid_idx)}")

            print(f"Indices in test_idx but not in split_idx['test']: {set(test_idx) - set(split_idx['test'])}")
            print(f"Indices in split_idx['test'] but not in test_idx: {set(split_idx['test']) - set(test_idx)}")

        print("="*20)
                        #print(len(test_idx))
            #print(len(split_idx['test']))
            #print(test_idx)
            #print(split_idx['test'])


    # First get smiles strings for the task


    #train_cutoff = floor(args.train_prop * len(all_idx))
    #valid_cutoff = int(np.round((1-args.train_prop)*1/2*len(all_idx)))


def test_custom_80_10_10_random_split_matches_ground_truth():
    # for better console readability.
    import warnings
    warnings.filterwarnings(
        "ignore",
        message="You are using `torch.load` with `weights_only=False`",
        category=FutureWarning
    )

    return_types = ["dgl_graph", "targets"]
    noise_level = 0.0
    device = torch.device("cuda:0")
    seed=1
    seed_data=123
    seed_all(seed)
    datasets = ["ogbg-molbace", "ogbg-molbbbp", "ogbg-molclintox", "ogbg-molmuv", "ogbg-molsider", "ogbg-moltox21", "ogbg-moltoxcast", "ogbg-molhiv", "ogbg-molesol", "ogbg-molfreesolv", "ogbg-mollipo"]
    #datasets = ["ogbg-molfreesolv"]
    
    for dataset_name in tqdm(datasets, desc="Processing datasets"):
        dataset = OGBGDatasetExtension(return_types=return_types, device=device, name=dataset_name, noise_level=noise_level)

        # ground truth
        split_idx = dataset.get_idx_split()
        all_idx = get_random_indices(len(dataset), seed_data)
        
        split_idx["train"] = all_idx[:len(split_idx["train"])]
        split_idx["valid"] = all_idx[len(split_idx["train"]):len(split_idx["train"])+len(split_idx["valid"])]
        split_idx["test"] = all_idx[len(split_idx["train"])+len(split_idx["valid"]):]

        # get custom splits
        custom_split = random_split(len_dataset=len(dataset),  frac_train=0.8, seed_data=seed_data)
        train_idx = custom_split['train']; valid_idx = custom_split['valid']; test_idx = custom_split['test']

        print("Train, valid and test contains all indices: ", len(set(list(train_idx) + list(valid_idx) + list(test_idx))-set(all_idx))==0)

        train_is_equal = all(train_idx == split_idx['train']) 
        valid_is_equal = all(valid_idx == split_idx['valid']) 
        test_is_equal = all(test_idx == split_idx['test']) 
        print(f"{dataset_name} - Train idx match: {train_is_equal}")
        print(f"{dataset_name} - Valid idx match: {valid_is_equal}")
        print(f"{dataset_name} - Test idx match:  {test_is_equal}")
        
        if train_is_equal == False or test_is_equal == False or test_is_equal == False:
            print(f"Indices in train_idx but not in split_idx['train']: {set(train_idx) - set(split_idx['train'])}")
            print(f"Indices in split_idx['train'] but not in train_idx: {set(split_idx['train']) - set(train_idx)}")

            print(f"Indices in valid_idx but not in split_idx['valid']: {set(valid_idx) - set(split_idx['valid'])}")
            print(f"Indices in split_idx['valid'] but not in valid_idx: {set(split_idx['valid']) - set(valid_idx)}")

            print(f"Indices in test_idx but not in split_idx['test']: {set(test_idx) - set(split_idx['test'])}")
            print(f"Indices in split_idx['test'] but not in test_idx: {set(split_idx['test']) - set(test_idx)}")

        print("="*20)

def test_custom_scaffold_splits_contain_all_molecules():
    # for better console readability.
    import warnings
    warnings.filterwarnings(
        "ignore",
        message="You are using `torch.load` with `weights_only=False`",
        category=FutureWarning
    )

    return_types = ["dgl_graph", "targets"]
    noise_level = 0.0
    device = torch.device("cuda:0")

    train_props = [0.8, 0.7, 0.6]
    datasets = ["ogbg-molbace", "ogbg-molbbbp", "ogbg-molclintox", "ogbg-molmuv", "ogbg-molsider", "ogbg-moltox21", "ogbg-moltoxcast", "ogbg-molhiv", "ogbg-molesol", "ogbg-molfreesolv", "ogbg-mollipo"]

    for dataset_name in tqdm(datasets):   
        dataset = OGBGDatasetExtension(return_types=return_types, device=device, name=dataset_name, noise_level=noise_level)

        all_idx = dataset.get_all_indices()

        for train_prop in train_props:
            custom_split =  scaffold_split(dataset_name, frac_train=train_prop)
            train_idx = custom_split['train']; valid_idx = custom_split['valid']; test_idx = custom_split['test']


            print(f"Dataset: {dataset_name}, train proportion: {train_prop}")
            print(f"Length of train: {len(train_idx)}\nLength of valid: {len(valid_idx)}\nLength of test: {len(test_idx)}")
            print("Train, valid and test contains all indices: ", len(set(train_idx + valid_idx + test_idx)-set(all_idx))==0, "\n")

def test_custom_random_splits_contain_all_molecules():
    # for better console readability.
    import warnings
    warnings.filterwarnings(
        "ignore",
        message="You are using `torch.load` with `weights_only=False`",
        category=FutureWarning
    )

    return_types = ["dgl_graph", "targets"]
    noise_level = 0.0
    device = torch.device("cuda:0")
    seed=1
    seed_data=123
    seed_all(seed)

    train_props = [0.8, 0.7, 0.6]
    datasets = ["ogbg-molbace", "ogbg-molbbbp", "ogbg-molclintox", "ogbg-molmuv", "ogbg-molsider", "ogbg-moltox21", "ogbg-moltoxcast", "ogbg-molhiv", "ogbg-molesol", "ogbg-molfreesolv", "ogbg-mollipo"]

    for dataset_name in tqdm(datasets):   
        dataset = OGBGDatasetExtension(return_types=return_types, device=device, name=dataset_name, noise_level=noise_level)

        all_idx = dataset.get_all_indices()

        for train_prop in train_props:
            custom_split = random_split(len_dataset=len(dataset),  frac_train=0.8, seed_data=seed_data)
            train_idx = custom_split['train']; valid_idx = custom_split['valid']; test_idx = custom_split['test']


            print(f"Dataset: {dataset_name}, train proportion: {train_prop}")
            print(f"Length of train: {len(train_idx)}\nLength of valid: {len(valid_idx)}\nLength of test: {len(test_idx)}")
            print("Train, valid and test contains all indices: ", len(set(list(train_idx) + list(valid_idx) + list(test_idx))-set(all_idx))==0, "\n")

def test_if_SMILES_in_3DInfomax_are_the_same_as_in_GraphMVP():
    """
    Test how many SMILES are shared in the different model's repositories

    Must mount container in Thesis instead of Thesis/3DInfomax to run this since we load smiles from GraphMVP
    """
    from rdkit import RDLogger   
    RDLogger.DisableLog('rdApp.*') 

    args = get_arguments()
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.device == 'cuda' else "cpu")

    datasets = ["ogbg-molbace", "ogbg-molbbbp", "ogbg-molclintox", "ogbg-molsider", "ogbg-moltox21", "ogbg-moltoxcast", "ogbg-molhiv", "ogbg-molesol", "ogbg-molfreesolv", "ogbg-mollipo"]
    #datasets = ["ogbg-molclintox"]

    for dataset in datasets:
        # 3DInfomax
        path_to_mol_info_3DInfomax = f"/workspace/3DInfomax/dataset/{dataset.replace('-','_')}/mapping/mol.csv.gz"

        df_3DInfomax = pd.read_csv(path_to_mol_info_3DInfomax, compression="gzip")
        df_smiles_3DInfomax = df_3DInfomax["smiles"].to_list()

        # GraphMVP
        dataset_naming_GraphMVP = dataset.replace("ogbg-mol", "", 1) 
        if dataset_naming_GraphMVP == 'lipo':
            dataset_naming_GraphMVP = 'lipophilicity'

        path_to_mol_info_GraphMVP = os.path.join('/workspace/GraphMVP/datasets/molecule_datasets', dataset_naming_GraphMVP, 'processed/smiles.csv')
        df_smiles_GraphMVP = pd.read_csv(path_to_mol_info_GraphMVP, header=None)[0].tolist()
    
        print(f"\n===SMILES of molecules in {dataset.replace('ogbg-mol', '', 1) }===")
        
        print(f'#SMILES in 3DInfomax: {len(df_smiles_3DInfomax)}')
        print(f'#SMILES in GraphMVP: {len(df_smiles_GraphMVP)}')
        print(f'#SMILES in common {len([smile for smile in df_smiles_3DInfomax if smile in df_smiles_GraphMVP])}')

        canonical_SMILES_in_common = []
        for i in range(len(df_3DInfomax)):
            mol1 = Chem.MolFromSmiles(df_smiles_3DInfomax[i])
            mol2 = Chem.MolFromSmiles(df_smiles_GraphMVP[i])
            canonical1 = Chem.MolToSmiles(mol1)
            canonical2 = Chem.MolToSmiles(mol2)
            canonical_SMILES_in_common.append(canonical1==canonical2)

        print(f'#canonical SMILES in common: {sum(canonical_SMILES_in_common)}\n')

if __name__ == "__main__":
    #test_scaffold_splits_for_different_models()
    #test_custom_80_10_10_scaffold_split_matches_ground_truth()
    #test_custom_80_10_10_random_split_matches_ground_truth()
    #test_custom_scaffold_splits_contain_all_molecules()
    #test_custom_random_splits_contain_all_molecules()
    test_if_SMILES_in_3DInfomax_are_the_same_as_in_GraphMVP()