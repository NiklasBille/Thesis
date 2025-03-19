import os
import torch

if __name__ == "__main__":
    path_to_data = "datasets/GEOM/GEOM_3D_nmol10_nconf5_nupper1000_morefeat/processed/geometric_data_processed.pt"

    data_obj, idx_boundaries = torch.load(path_to_data)

    print("======= Data object =======\n")
    print(data_obj, "\n")

    print("======= Dict of index boundaries =======\n")
    print(idx_boundaries)
    print(len(idx_boundaries["x"]))
    #print(data[0].mol_id)
