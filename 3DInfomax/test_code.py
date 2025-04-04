import os
import sys
from collections import Counter

import pandas as pd
import torch
import numpy as np

def test_classification_unique_classes():
    """
    Find out which classification datasets contain missing labels since these require
    the loss function OGBNanLabelBCEWithLogitsLoss.
    """

    classification_datasets = ["ogbg_molbace", "ogbg_molbbbp", "ogbg_molclintox", "ogbg_molmuv", "ogbg_molsider", "ogbg_moltox21", "ogbg_moltoxcast", "ogbg_molhiv"]

    for dataset in classification_datasets:

        path_to_processed_data = os.path.join("/workspace","dataset", dataset, "processed/") + str("data_processed")

        loaded_dict = torch.load(str(path_to_processed_data), 'rb')

        classes = set()
        label_counter = Counter()

        nan_found = False
        for graph_labels in loaded_dict['labels']:

            for label in graph_labels:
                if np.isnan(label):
                    label_counter["nan"] += 1
                else:
                    label_counter[label] += 1

            for label in np.unique(graph_labels):

                if np.isnan(label): # since np.nan can appear multiple times in a set we only insert it first time
                    if not nan_found:
                        nan_found = True
                        classes.add(label)
                else:
                    classes.add(label)    
        
        print(f"Unique labels in {dataset}: {list(classes)}")
        print(f"Occurences of classes in {dataset}: {label_counter}")
        print(f"Number of molecules in {dataset}: {len(loaded_dict['labels'])}\n")




if __name__ == "__main__":
    test_classification_unique_classes()