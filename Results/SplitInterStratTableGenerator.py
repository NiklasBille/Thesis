import sys
from typing_extensions import override #to explicitly state when overriding method
import pandas as pd
import numpy as np

import RawTableGenerator as tg


class SplitInterStratTableGenerator(tg.RawTableGenerator):
    def __init__(self, model, experiment, partition, decimals=None):
        super().__init__(model, experiment, partition, decimals)
        self.model=model
        self.raw_primary_table, self.raw_secondary_table = super().create_table(experiment, model, partition)

    def _compute_delta_table(self, df, use_secondary_metric=False):
        # Create empty table to hold delta values
        #columns = ["metric", "train_prop=0.8", "train_prop=0.7", "train_prop=0.6"]
        #delta_table = pd.DataFrame(index=self.datasets, columns=columns)

        train_props = ["train_prop=0.8", "train_prop=0.7", "train_prop=0.6"]
        columns = pd.MultiIndex.from_product(
            [train_props, ["mean", "std"]],
            names=["train_prop", "stat"]
        )
    
        delta_table = pd.DataFrame(index=self.datasets, columns=columns)

        for dataset, row in df.iterrows():
            # First compute the delta value
            R_random = row.loc['random']; R_scaffold = row.loc['scaff']
            R_random_mean = R_random[:, "mean"]; R_scaffold_mean = R_scaffold[:, "mean"]
            R_random_std = R_random[:, "std"]; R_scaffold_std = R_scaffold[:, "std"]
            
            delta = (R_scaffold_mean - R_random_mean)

            # Then approximate the standard deviation of delta for each train_prop
            for i, tp in enumerate(train_props):
                results_random = self.extract_results_split(model=self.model, dataset=dataset, split_strategy='random', train_prop=tp)
                results_scaffold = self.extract_results_split(model=self.model, dataset=dataset, split_strategy='scaff', train_prop=tp)

                if dataset == 'bbbp' and tp == 'train_prop=0.6':
                    continue

                if dataset in ['freesolv', 'esol', 'lipo']:
                    metric = 'mae' if use_secondary_metric else 'rmse'       
                    X = results_scaffold[f'{self.partition}_{metric}']
                    Y = results_random[f'{self.partition}_{metric}']                         
                else:
                    metric = 'prcauc' if use_secondary_metric else 'rocauc'
                    X = [100*result for result in results_scaffold[f'{self.partition}_{metric}']]
                    Y = [100*result for result in results_random[f'{self.partition}_{metric}']]

                n = len(X)
                sigma2_X = np.var(X, ddof=1)
                sigma2_Y = np.var(Y, ddof=1)                
                cov = np.cov(X,Y, ddof=1)[0, 1]
                approx_std = np.sqrt(sigma2_X/n + sigma2_Y/n - 2*cov/n)
                
                delta_table.loc[dataset, (tp, 'mean')] = delta[i]
                delta_table.loc[dataset, (tp, 'std')] = approx_std

        delta_table.insert(loc=0, column='metric', value=df['metric'])

        return delta_table


    @override
    def create_table(self):
        primary_table = self._compute_delta_table(self.raw_primary_table)
        secondary_table = self._compute_delta_table(self.raw_secondary_table, use_secondary_metric=True)

        return primary_table, secondary_table

    @override
    def print_result_table(self, print_secondary_metric=False):
        print("\n" + "="*80)
        print(f"MODEL: {self.model} | EXPERIMENT: {self.experiment} | PARITION: {self.partition} | Inter-strategy compare")
        print("="*80)
        primary_table, secondary_table = self.create_table()
        if self.decimals is not None:
            self.round_table(primary_table)
            self.round_table(secondary_table)
        if print_secondary_metric is False:
            print("PRIMARY METRIC DELTA TABLE")
            print(primary_table.to_string())
            print("\n" + "-"*80)
        else:
            print("SECONDARY METRIC DELTA TABLE")
            print(secondary_table.to_string())
            print("\n" + "-"*80)

    @override
    def convert_table_to_latex(self, print_secondary_metric=False):
        table_primary_metric, table_secondary_metric = self.create_table()
        table = table_primary_metric if print_secondary_metric is False else table_secondary_metric

        # Create a table to hold "mean \pm std"
        combined = pd.DataFrame(index=table.index)

        # Dictionaries for rename mapping
        metric_rename = {"rmse": "RMSE $\\downarrow$", "mae": "MAE $\\downarrow$", "rocauc": "ROC-AUC $\\uparrow$", "prcauc": "PRC-AUC $\\uparrow$"}
        dataset_rename = {"freesolv": "FreeSolv", "esol": "ESOL", "lipo": "Lipo", "bace": "BACE", "bbbp": "BBBP", "clintox": "ClinTox", "hiv": "HIV", "muv": "MUV", "sider": "SIDER", "toxcast": "ToxCast", "tox21": "Tox21"}

        train_props = ["train_prop=0.8", "train_prop=0.7", "train_prop=0.6"]

        # First populate the table
        for train_prop in train_props:
        
            mean_col = (train_prop, "mean")
            std_col = (train_prop, "std")

            combined[train_prop] = table.apply(
                lambda row: f"{row[mean_col]:.{self.decimals}f}$\\pm${row[std_col]:.{self.decimals}f}" 
                            if pd.notna(row[mean_col]) and pd.notna(row[std_col]) 
                            else "*",
                axis=1
            )

        # Then rename columns and rows accordingly
        rename_mapping = {f"train_prop={p}": f"$p={p}$" for p in [0.8, 0.7, 0.6]}
        combined.rename(columns=rename_mapping, inplace=True)
        combined.rename(index=dataset_rename, inplace=True)

        latex_str = combined.to_latex()
        print(latex_str)

        