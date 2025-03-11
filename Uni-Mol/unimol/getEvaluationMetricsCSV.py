import sys
import os

import pandas as pd
import numpy as np
from math import sqrt
from sklearn.metrics import roc_auc_score

def get_csv_results_reg(predict_path, csv_path, writing_to_csv):
    """
    Construct both mean aggregated df and regular df of smi_name, predict value and target value for a regression task.
    Aggregation is based on smi_name.
    """
    predict = pd.read_pickle(predict_path)
    smi_list, predict_list, target_list = [], [], []
    for batch in predict:
        sz = batch["bsz"]
        for i in range(sz):
            yhat = batch["predict"][i].cpu()
            y = batch["target"][i].cpu()
            
            smi_list.append(batch["smi_name"][i])
            predict_list.append(yhat.detach().item())
            target_list.append(y.detach().item())
            
    predict_df = pd.DataFrame({"SMILES": smi_list, "predict": predict_list, "target": target_list})
    predict_df_agg = predict_df.groupby("SMILES").mean()

    if writing_to_csv:
        predict_df.to_csv(csv_path,index=False)
        predict_df_agg.to_csv(csv_path.replace("test", "test_agg"),index=False)

    return predict_df, predict_df_agg

if __name__ == "__main__":

    WRITE_TO_CSV = True 

    #datasets = ["freesolv_seed0", "freesolv_seed3", "freesolv_seed4"]
    #datasets = ["qm7_seed2", "qm7_seed3", "qm7_seed4"]
    #datasets = ["esol_seed2", "esol_seed3", "esol_seed4"]
    datasets = ["lipo_seed2", "lipo_seed3", "lipo_seed4"]

    RMSE_list = []
    RMSE_agg_list = []
    MAE_list = []
    MAE_agg_list = []

    print("\n===== Dataset Metrics =====")
    
    for dataset in datasets:
        predict_path = f"results/{dataset}_test.out.pkl"
        csv_path = f"results/{dataset}_test.out.csv"
        predict_df, predict_df_agg = get_csv_results_reg(predict_path, csv_path, WRITE_TO_CSV)
        
        print(f"\nDataset: {dataset}")
        rmse = sqrt(((predict_df['predict'] - predict_df['target']) ** 2).mean())
        rmse_agg = sqrt(((predict_df_agg['predict'] - predict_df_agg['target']) ** 2).mean())
            
        print(f"  - RMSE: {rmse:.3f}")
        print(f"  - Aggregated RMSE: {rmse_agg:.3f}")
        
        RMSE_list.append(rmse)
        RMSE_agg_list.append(rmse_agg)

        mae = np.abs(predict_df['predict'] - predict_df['target']).mean()
        mae_agg = np.abs(predict_df_agg['predict'] - predict_df_agg['target']).mean()
        
        print(f"  - MAE: {mae:.3f}")
        print(f"  - Aggregated MAE: {mae_agg:.3f}")

        MAE_list.append(mae)
        MAE_agg_list.append(mae_agg)
    
    print("\n===== Overall Metrics =====")
    print(f"Average RMSE: {np.mean(RMSE_list):.3f}")
    print(f"Standard Deviation (RMSE): {np.std(RMSE_list):.3f}")
    print(f"Average RMSE (Aggregated): {np.mean(RMSE_agg_list):.3f}")
    print(f"Standard Deviation (Aggregated RMSE): {np.std(RMSE_agg_list):.3f}\n")
    
    print(f"Average MAE: {np.mean(MAE_list):.3f}")
    print(f"Standard Deviation (MAE): {np.std(MAE_list):.3f}")
    print(f"Average MAE (Aggregated): {np.mean(MAE_agg_list):.3f}")
    print(f"Standard Deviation (Aggregated MAE): {np.std(MAE_agg_list):.3f}\n")