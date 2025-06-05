import sys
import RawTableGenerator as tg
import pandas as pd
import numpy as np

from typing import Literal
from typing_extensions import override #to explicitly state when overriding method

def approx_variance_delta_method(X, Y):
    n = len(X)   

    mu_X = np.mean(X)
    mu_Y = np.mean(Y)
    sigma2_X = np.var(X, ddof=1)/n
    sigma2_Y = np.var(Y, ddof=1)/n
    cov = np.cov(X, Y, ddof=1)[0,1]/n
    
    return 100**2 * (sigma2_X/(mu_Y**2) + mu_X**2 * sigma2_Y/(mu_Y**4) - 2 * mu_X/(mu_Y**3)*cov)

class ModelComparisonTableGenerator(tg.RawTableGenerator):
    def __init__(self, experiment, partition, list_of_models, decimals=None):
        super().__init__(experiment=experiment, partition=partition, decimals=decimals, isComparingModels=True)
        self.list_of_models = list_of_models
        self.raw_table_dict = dict.fromkeys(self.list_of_models)
        for model in self.list_of_models:
            raw_primary_table, raw_secondary_table =  super().create_table(experiment, model, partition)
            self.raw_table_dict[model] = {'primary': raw_primary_table, 'secondary': raw_secondary_table}
    
    def set_table_dict(self, table_dict):
        """
        Can be used to inject precomputed metric tables for comparisson
        """
        self.raw_table_dict = table_dict

    def _compute_model_comparison_table(self, raw_table_dict, metric_type: Literal['primary', 'secondary']):

        # Create empty MultiIndex table
        if self.experiment == "noise":
            possible_sub_experiments = ["noise=0.0", "noise=0.05", "noise=0.1", "noise=0.2"] 
            columns = pd.MultiIndex.from_product(
                [possible_sub_experiments, self.list_of_models, ["mean", "std"]],
                names=["sub_experiment", "model", "stat"]
                )
        else:
            possible_sub_experiments = ["random", "scaff"]
            train_props = ["train_prop=0.8", "train_prop=0.7", "train_prop=0.6"]
            columns = pd.MultiIndex.from_product(
                [possible_sub_experiments, train_props, self.list_of_models, ["mean", "std"]],
                names=["sub_experiment", "train_prop", "model", "stat"]
            )
            
        table = pd.DataFrame(index=self.datasets, columns=columns)

        # Insert a column for information on metrics
        first_entry_key = list(raw_table_dict)[0]
        table.insert(0, 'metric', raw_table_dict[first_entry_key][metric_type]['metric'])

        # Fill table
        for model, raw_tables in raw_table_dict.items():
            raw_table = raw_tables[metric_type]
            
            # The table for each experiment is different, so they are processed in different ways
            if self.experiment == 'noise':
                # Add each sub experiment metrics to the correct colunm
                for sub_experiment in possible_sub_experiments:
                    R = raw_table.loc[self.datasets, sub_experiment]
                    table.loc[:, (sub_experiment, model, "mean")] = R["mean"]
                    table.loc[:, (sub_experiment, model, "std")] = R["std"]

            elif self.experiment =='split':
                # Add each sub experiment metrics to the correct colunm
                for sub_experiment in possible_sub_experiments:
                    for train_prop in train_props:
                        R = raw_table.loc[self.datasets, (sub_experiment, train_prop)]
                        table.loc[:, (sub_experiment, train_prop, model, "mean")] = R["mean"]
                        table.loc[:, (sub_experiment, train_prop, model, "std")] = R["std"]

        return table

        

    @override
    def create_table(self):
        primary_table = self._compute_model_comparison_table(self.raw_table_dict, 'primary')
        secondary_table = self._compute_model_comparison_table(self.raw_table_dict, 'secondary')

        return primary_table, secondary_table
    
    @override
    def print_result_table(self, print_secondary_metric=False, use_percentage=False):
        print("\n" + "="*80)
        if use_percentage is True:
            print(f"EXPERIMENT: {self.experiment} | PARTITION: {self.partition} | Comparing models | Change in percentage")
        else:
            print(f"EXPERIMENT: {self.experiment} | PARTITION: {self.partition} | Comparing models")
        print("="*80)
        primary_table, secondary_table = self.create_table()
        if self.decimals is not None:
            self.round_table(primary_table)
            self.round_table(secondary_table)
    
        if self.experiment == 'noise':
            if print_secondary_metric is False:
                print("PRIMARY METRIC")
                print(primary_table.to_string())
                print("\n" + "-"*80)
            else:
                print("SECONDARY METRIC")
                print(secondary_table.to_string())
                print("\n" + "-"*80)
            
        # Split table is very large so we present it as two tables, one for random and one for scaffold
        else: 
            if print_secondary_metric is False:
                print("PRIMARY METRIC")
                print("[RANDOM]")
                primary_table.set_index('metric', append=True, inplace=True) # keep metric column when slicing
                print(primary_table.loc[:, 'random'].to_string(), '\n')
                print("[SCAFFOLD]")
                print(primary_table.loc[:, 'scaff'].to_string(), '\n', '-'*80)

            else:
                secondary_table.set_index('metric', append=True, inplace=True) # keep metric column when slicing
                print("SECONDARY METRIC")
                print("[RANDOM]")
                print(secondary_table.loc[:, 'random'].to_string(), '\n')
                print("[SCAFFOLD]")
                print(secondary_table.loc[:, 'scaff'].to_string(), '\n', '-'*80)

    def compute_std_noise(self, dataset, model, noise_level, partition, use_secondary_metric):
        # First extract baseline results
        baseline_noise = 'noise=0.0'
        baseline_results = self.extract_results_noise(model, dataset, baseline_noise)
        # Then extract readout results
        readout_results = self.extract_results_noise(model, dataset, noise_level)
        
        if dataset in ['freesolv', 'esol', 'lipo']:
            metric = 'mae' if use_secondary_metric else 'rmse' 
            X = readout_results[f'{partition}_{metric}']
            Y = baseline_results[f'{partition}_{metric}']                               
        else:
            metric = 'prcauc' if use_secondary_metric else 'rocauc'
            X = [100*result for result in readout_results[f'{partition}_{metric}']]
            Y = [100*result for result in baseline_results[f'{partition}_{metric}']]

                                                                
        std_noise = np.sqrt(approx_variance_delta_method(X, Y))

        #TODO figure out why this variance can be negative for HIV?

        return std_noise
    
    def compute_std_split(self, dataset, model, split_strategy, train_prop, use_secondary_metric):
        # First extract baseline results
        baseline_split = 'train_prop=0.8'
        baseline_results = self.extract_results_split(model, dataset, split_strategy, baseline_split)
        # Then extract readout results
        readout_results = self.extract_results_split(model, dataset, split_strategy, train_prop)

        if dataset in ['freesolv', 'esol', 'lipo']:
            metric = 'mae' if use_secondary_metric else 'rmse'                                
        else:
            metric = 'prcauc' if use_secondary_metric else 'rocauc'

        X = readout_results[f'{self.partition}_{metric}']
        Y = baseline_results[f'{self.partition}_{metric}']

        std_noise = np.sqrt(approx_variance_delta_method(X, Y))

        return std_noise

    @override
    def convert_table_to_latex(self, print_secondary_metric=False, use_percentage=False):
        primary_table, secondary_table = self.create_table()
        table = primary_table if print_secondary_metric is False else secondary_table

         # Create a table to hold "mean \pm std"
        mean_columns = table.columns.get_level_values('stat') == 'mean'

        if self.experiment == "noise":
            columns_without_stat = table.columns[mean_columns].droplevel('stat').set_names([None, None])
        elif self.experiment == "split":
            columns_without_stat = table.columns[mean_columns].droplevel('stat')

        combined = pd.DataFrame(index=table.index, columns=columns_without_stat)

        # Dictionaries for rename mapping
        metric_rename = {"rmse": "RMSE $\\downarrow$", "mae": "MAE $\\downarrow$", "rocauc": "ROC-AUC $\\uparrow$", "prcauc": "PRC-AUC $\\uparrow$"}
        dataset_rename = {"freesolv": "FreeSolv $\\downarrow$", "esol": "ESOL $\\downarrow$", "lipo": "Lipo $\\downarrow$", "bace": "BACE $\\uparrow$", "bbbp": "BBBP $\\uparrow$", "clintox": "ClinTox $\\uparrow$", "hiv": "HIV $\\uparrow$", "muv": "MUV $\\uparrow$", "sider": "SIDER $\\uparrow$", "toxcast": "ToxCast $\\uparrow$", "tox21": "Tox21 $\\uparrow$"}
        # For noise experiment
        if self.experiment == "noise":
            possible_sub_experiments = ["noise=0.0", "noise=0.05", "noise=0.1", "noise=0.2"] 
            
            # Populate the table
            for sub_exp in possible_sub_experiments:
                for model in self.list_of_models:
                    mean_col = (sub_exp, model, "mean")
                    std_col = (sub_exp, model, "std")
                    
                    # Baseline should always show raw numbers + standard deviation regardless if we use_percentage
                    if sub_exp == "noise=0.0" or not use_percentage:
                        combined[sub_exp, model] = table.apply(
                            lambda row: f"{row[mean_col]:.{self.decimals}f}$\\pm${row[std_col]:.{self.decimals}f}" 
                                        if pd.notna(row[mean_col]) and pd.notna(row[std_col]) 
                                        else pd.NA,
                            axis=1
                        )
                        # combined[sub_exp, model] = table.apply(
                        #     lambda row: (
                        #         f"{row[mean_col]:.{self.decimals + 1}f}$\\pm${row[std_col]:.{self.decimals+1}f}" 
                        #         if row.name in ["freesolv", "esol", "lipo"]
                        #         else (
                        #             f"{row[mean_col]:.{self.decimals}f}$\\pm${row[std_col]:.{self.decimals}f}"
                        #             if pd.notna(row[mean_col]) and pd.notna(row[std_col])
                        #             else pd.NA
                        #         )
                        #     ),
                        #     axis=1
                        # )        
                    else:
                        combined[sub_exp, model] = table.apply(
                            lambda row: f"{row[mean_col]:.{self.decimals}f}\pct$\\pm${self.compute_std_noise(dataset=row.name, model=model, noise_level=sub_exp, partition=self.partition, use_secondary_metric=print_secondary_metric):.{self.decimals}f}"
                                        if pd.notna(row[mean_col]) 
                                        else pd.NA,
                            axis=1
                        )
            
            # Rename columns
            rename_mapping = {f"noise={p}": f"$\eta={p}$" for p in [0.0, 0.05, 0.1, 0.2]}
            combined.rename(columns=rename_mapping, inplace=True)
            # Add metric column back
            # combined.insert(0, column=('', 'Metric'), value=table["metric"])
            # combined["","Metric"] = combined["","Metric"].replace(metric_rename)
            combined.rename(index=dataset_rename, inplace=True)

            latex_str = combined.to_latex()
            latex_str = latex_str.replace("GraphCL_1", "GraphCL").replace("GraphCL_2", "GraphCL")
            latex_str = latex_str.replace(r'\multicolumn{2}{r}', r'\multicolumn{2}{c}')

            # Define title row (spanning all columns)
            title = rf"\toprule" + "\n" + rf"\multicolumn{{9}}{{c}}{{\textbf{{Noise experiment}}}} \\" + "\n" + r"\midrule"

            # Inject the title after \toprule and before the column headers
            latex_str = latex_str.replace(r"\toprule", title, 1)

            # Wrap entire str in a resize box
            latex_str = "\\resizebox{\\textwidth}{!}{%\n" + latex_str + "}"
            print(latex_str)
        
        elif self.experiment == "split":
            possible_sub_experiments = ["random", "scaff"]
            train_props = ["train_prop=0.8", "train_prop=0.7", "train_prop=0.6"]

            # First populate the table
            for sub_exp in possible_sub_experiments:
                for train_prop in train_props:
                    for model in self.list_of_models:
                        mean_col = (sub_exp, train_prop, model, "mean")
                        std_col = (sub_exp, train_prop, model, "std")
            
                        if train_prop == "train_prop=0.8" or not use_percentage:
                            combined[sub_exp, train_prop, model] = table.apply(
                                lambda row: f"{row[mean_col]:.{self.decimals}f}$\\pm${row[std_col]:.{self.decimals}f}" 
                                            if pd.notna(row[mean_col]) and pd.notna(row[std_col]) 
                                            else "*",
                                axis=1
                            )
                            # combined[sub_exp, train_prop, model] = table.apply(
                            #     lambda row: (
                            #         f"{row[mean_col]:.{self.decimals + 1}f}$\\pm${row[std_col]:.{self.decimals+1}f}" 
                            #         if row.name in ["freesolv", "esol", "lipo"]
                            #         else (
                            #             f"{row[mean_col]:.{self.decimals}f}$\\pm${row[std_col]:.{self.decimals}f}"
                            #             if pd.notna(row[mean_col]) and pd.notna(row[std_col])
                            #             else pd.NA
                            #         )
                            #     ),
                            #     axis=1
                            # )        
                        else:
                            combined[sub_exp, train_prop, model] = table.apply(
                                lambda row: f"{row[mean_col]:.{self.decimals}f}\pct$\\pm${self.compute_std_split(dataset=row.name, model=model, split_strategy=sub_exp, train_prop=train_prop, use_secondary_metric=print_secondary_metric):.{self.decimals}f}"
                                            if pd.notna(row[mean_col]) 
                                            else "*",
                                axis=1
                            )
            
            # Extract columns
            random_cols = [col for col in combined.columns if isinstance(col, tuple) and col[0] == 'random']
            scaff_cols = [col for col in combined.columns if isinstance(col, tuple) and col[0] == 'scaff']

            # Create new tables 
            combined_random = combined[random_cols]
            combined_scaff = combined[scaff_cols]

            # Rename columns
            combined_random.columns =  pd.MultiIndex.from_tuples([(f"$p={col[1].split('=')[1]}$", col[2]) for col in combined_random.columns if isinstance(col, tuple) and len(col) == 3], names=["train_prop", "model"])
            combined_scaff.columns =  pd.MultiIndex.from_tuples([(f"$p={col[1].split('=')[1]}$", col[2]) for col in combined_scaff.columns if isinstance(col, tuple) and len(col) == 3], names=["train_prop", "model"])

            # Add metric column back
            #combined_random.insert(0, column=('', 'Metric'), value=table["metric"])
            #combined_scaff.insert(0, column=('', 'Metric'), value=table["metric"])
            
         
            # Rename metrics
            combined_random = combined_random.copy() # to suppres a warning
            # combined_random[("", "Metric")] = combined_random[("", "Metric")].replace(metric_rename)

            combined_scaff = combined_scaff.copy() # to suppres a warning
            # combined_scaff[("", "Metric")] = combined_scaff[("", "Metric")].replace(metric_rename)

            combined_random.rename(index=dataset_rename, inplace=True)
            combined_scaff.rename(index=dataset_rename, inplace=True)

            combined_random.columns.names = [None, None]  # remove 'train_prop' and 'model' names
            combined_scaff.columns.names = [None, None]  # remove 'train_prop' and 'model' names
            
            latex_str_random = combined_random.to_latex()
            latex_str_scaff = combined_scaff.to_latex()

            latex_str_random = latex_str_random.replace("GraphCL_1", "GraphCL").replace("GraphCL_2", "GraphCL")
            latex_str_scaff = latex_str_scaff.replace("GraphCL_1", "GraphCL").replace("GraphCL_2", "GraphCL")
            latex_str_random = latex_str_random.replace(r'\multicolumn{2}{r}', r'\multicolumn{2}{c}')
            latex_str_scaff = latex_str_scaff.replace(r'\multicolumn{2}{r}', r'\multicolumn{2}{c}')

            # Define title row (spanning all columns)
            title_random = rf"\toprule" + "\n" + rf"\multicolumn{{7}}{{c}}{{\textbf{{Random splits}}}} \\" + "\n" + r"\midrule"
            title_scaff = rf"\toprule" + "\n" + rf"\multicolumn{{7}}{{c}}{{\textbf{{Scaffold splits}}}} \\" + "\n" + r"\midrule"

            # Inject the title after \toprule and before the column headers
            latex_str_random = latex_str_random.replace(r"\toprule", title_random, 1)
            latex_str_scaff = latex_str_scaff.replace(r"\toprule", title_scaff, 1)

            # Wrap entire str in a resize box
            latex_str_random = "\\resizebox{\\textwidth}{!}{%\n" + latex_str_random + "}"
            latex_str_scaff = "\\resizebox{\\textwidth}{!}{%\n" + latex_str_scaff + "}"
            print(latex_str_random, "\n")
            print(latex_str_scaff)


class ModelComparisonSplitInterStratTableGenerator(ModelComparisonTableGenerator):
    def __init__(self, experiment, partition, list_of_models, decimals=None):
        super().__init__(experiment=experiment, partition=partition, list_of_models=list_of_models, decimals=decimals)


    @override
    def print_result_table(self, print_secondary_metric=False):
        print("\n" + "="*80)
        print(f"EXPERIMENT: {self.experiment} | PARTITION: {self.partition} | Comparing models")
        print("="*80)
        primary_table, secondary_table = self.create_table()
        if self.decimals is not None:
            self.round_table(primary_table)
            self.round_table(secondary_table)

        if print_secondary_metric is False:
            print("PRIMARY METRIC")
            print(primary_table.to_string())
            print("\n" + "-"*80)
        else:
            print("SECONDARY METRIC")
            print(secondary_table.to_string())
            print("\n" + "-"*80)

    @override
    def _compute_model_comparison_table(self, raw_table_dict, metric_type: Literal['primary', 'secondary']):

        # Create empty MultiIndex table
        train_props = ["train_prop=0.8", "train_prop=0.7", "train_prop=0.6"]
        columns = pd.MultiIndex.from_product(
            [train_props, self.list_of_models, ["mean", "std"]],
            names=["train_prop", "model", "stat"]
        )
        
        table = pd.DataFrame(index=self.datasets, columns=columns)

        # Insert a column for information on metrics
        first_entry_key = list(raw_table_dict)[0]
        table.insert(0, 'metric', raw_table_dict[first_entry_key][metric_type]['metric'])

        # Fill table
        for model, raw_tables in raw_table_dict.items():
            raw_table = raw_tables[metric_type]
            for train_prop in train_props:
                R = raw_table.loc[self.datasets, (train_prop)]
                table.loc[:, (train_prop, model, "mean")] = R["mean"]
                table.loc[:, (train_prop, model, "std")] = R["std"]

        return table
    
    @override
    def convert_table_to_latex(self, print_secondary_metric=False):
        table_primary_metric, table_secondary_metric = self.create_table()
        table = table_primary_metric if print_secondary_metric is False else table_secondary_metric

        mean_columns = table.columns.get_level_values('stat') == 'mean'
        columns_without_stat = table.columns[mean_columns].droplevel('stat')

        combined = pd.DataFrame(index=table.index, columns=columns_without_stat)

        train_props = ["train_prop=0.8", "train_prop=0.7", "train_prop=0.6"]
        # First populate the table
        for train_prop in train_props:
            for model in self.list_of_models:
                mean_col = (train_prop, model, "mean")
                std_col = (train_prop, model, "std")
    
                combined[train_prop, model] = table.apply(
                    lambda row: f"{row[mean_col]:.{self.decimals}f}$\\pm${row[std_col]:.{self.decimals}f}" 
                                if pd.notna(row[mean_col]) and pd.notna(row[std_col]) 
                                else "*",
                    axis=1
                )

        rename_mapping = {f"train_prop={p}": f"$p={p}$" for p in [0.8, 0.7, 0.6]}
        combined.rename(columns=rename_mapping, inplace=True)

        dataset_rename = {"freesolv": "FreeSolv $\\downarrow$", "esol": "ESOL $\\downarrow$", "lipo": "Lipo $\\downarrow$", "bace": "BACE $\\uparrow$", "bbbp": "BBBP $\\uparrow$", "clintox": "ClinTox $\\uparrow$", "hiv": "HIV $\\uparrow$", "muv": "MUV $\\uparrow$", "sider": "SIDER $\\uparrow$", "toxcast": "ToxCast $\\uparrow$", "tox21": "Tox21 $\\uparrow$"}
        combined.rename(index=dataset_rename, inplace=True)
            
        combined.columns.names = [None, None] # Get rid of 'train_prop' and 'model' naming

        latex_str = combined.to_latex()
        latex_str = latex_str.replace("GraphCL_1", "GraphCL").replace("GraphCL_2", "GraphCL")

        latex_str = latex_str.replace(rf'\multicolumn{{{len(self.list_of_models)}}}{{r}}', rf'\multicolumn{{{len(self.list_of_models)}}}{{c}}')

        # Wrap entire str in a resize box
        latex_str = "\\resizebox{\\textwidth}{!}{%\n" + latex_str + "}"
        print(latex_str)
