import sys
import os

import pandas as pd
import numpy as np
from math import sqrt
from sklearn.metrics import roc_auc_score

def get_csv_results_reg(predict_path, csv_path):
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

    #print(f"Aggregated RMSE: {sqrt(((predict_df_agg['predict'] - predict_df_agg['target']) ** 2).mean()):.4}")

    predict_df.to_csv(csv_path,index=False)
    predict_df_agg.to_csv(csv_path.replace("test", "test_agg"),index=False)

    return predict_df, predict_df_agg

if __name__ == "__main__":

    datasets = ["freesolv"]

    for dataset in datasets:
        predict_path = f"results/{dataset}_test.out.pkl"
        csv_path = f"results/{dataset}_test.out.csv"
        predict_df, predict_df_agg = get_csv_results_reg(predict_path, csv_path)
        if dataset not in ["qm7", "qm7_no_hydrogen"] : 
            print(f"{dataset} RMSE: {sqrt(((predict_df['predict'] - predict_df['target']) ** 2).mean()):.4}")
            print(f"{dataset} Aggregated RMSE: {sqrt(((predict_df_agg['predict'] - predict_df_agg['target']) ** 2).mean()):.4}")
        else:
            print(f"{dataset} MAE: {np.abs(predict_df['predict'] - predict_df['target']).mean():.4}")
            print(f"{dataset} Aggregated MAE: {np.abs(predict_df_agg['predict'] - predict_df_agg['target']).mean():.4}")