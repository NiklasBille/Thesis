import numpy as np
import pandas as pd
import os
import sys
import warnings

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning) # To supress an annoying warning

class RawTableGenerator:
    def __init__(self, model=None, experiment=None, partition=None, decimals=None, isComparingModels=False):
        self.model = model
        self.experiment = experiment
        self.partition = partition
        self.decimals = decimals
        self.isComparingModels = isComparingModels    # Flag is only True when using model_comparison scripts
        self.datasets = ["freesolv", "esol", "lipo", "bace", "bbbp", "clintox", "hiv", "muv", "sider", "toxcast", "tox21"]
        self.allowed_experiments = ["noise", "split"]
        self.allowed_models = ["3DInfomax", "GraphMVP", "GraphCL_1", "GraphCL_2"]
        self.allowed_partitions = ["train", "val", "test"]

        self.validate_inputs()
    
    def validate_inputs(self):
        if self.experiment not in self.allowed_experiments:
            raise ValueError(f"Invalid experiment '{self.experiment}'. Must be one of {self.allowed_experiments}")
        
        if self.model not in self.allowed_models and not self.isComparingModels:
            raise ValueError(f"Invalid model '{self.model}'. Must be one of {self.allowed_models}")
        
        
    def get_dataset_task_type(self, dataset):
        dataset_task_types = {
        'esol': 'regression',
        'freesolv': 'regression',
        'lipo': 'regression',
        'hiv': 'classification',
        'muv': 'classification',
        'bace': 'classification',
        'bbbp': 'classification',
        'tox21': 'classification',
        'toxcast': 'classification',
        'sider': 'classification',
        'clintox': 'classification'
        }
        return dataset_task_types[dataset]

    def print_result_table(self, print_secondary_metric=False):
        print("\n" + "="*80)
        print(f"MODEL: {self.model} | EXPERIMENT: {self.experiment} | PARTITION: {self.partition}")
        print("="*80)
        table_primary_metric, table_secondary_metric = self.create_table(self.experiment, self.model, self.partition)
        if self.decimals is not None:
            self.round_table(table_primary_metric)
            self.round_table(table_secondary_metric)
        print("PRIMARY METRIC")
    
        if self.experiment == 'noise':
            print(table_primary_metric.to_string())
            print("\n" + "-"*80)
            if print_secondary_metric:
                print("\n SECONDARY METRIC")
                print(table_secondary_metric.to_string())
                print("\n" + "-"*80)
            
        # Split table is very large so we present it as two tables, one for random and one for scaffold
        else: 
            print("[RANDOM]")
            table_primary_metric.set_index('metric', append=True, inplace=True) # keep metric column when slicing
            print(table_primary_metric.loc[:, 'random'].to_string(), '\n')
            print("[SCAFFOLD]")
            print(table_primary_metric.loc[:, 'scaff'].to_string(), '\n', '-'*80)

            if print_secondary_metric:
                table_secondary_metric.set_index('metric', append=True, inplace=True) # keep metric column when slicing
                print("SECONDARY METRIC")
                print("[RANDOM]")
                print(table_secondary_metric.loc[:, 'random'].to_string(), '\n')
                print("[SCAFFOLD]")
                print(table_secondary_metric.loc[:, 'scaff'].to_string(), '\n', '-'*80)

        
    
    def round_table(self, table):
         if self.decimals is not None:
            # Get all columns except 'metric'
            value_columns = table.columns.drop("metric")
            
            # Convert those columns to float and round them
            table[value_columns] = (
                table[value_columns]
                .apply(pd.to_numeric, errors="coerce")  # safely convert to float
                .round(decimals=self.decimals)  
            )

    def get_task_type_metrics(self, task_type):
        primary_metric = "rmse" if task_type == "regression" else "rocauc"
        secondary_metric = "mae" if task_type == "regression" else "prcauc"
        return primary_metric, secondary_metric
    
    def create_table(self, experiment, model, partition):
        # Create empty MultiIndex table
        if experiment == "noise":
            possible_sub_experiments = ["noise=0.0", "noise=0.05", "noise=0.1", "noise=0.2"] 
            columns = pd.MultiIndex.from_product(
                [possible_sub_experiments, ["mean", "std"]],
                names=["sub_experiment", "stat"]
                )
        else:
            possible_sub_experiments = ["random", "scaff"]
            train_props = ["train_prop=0.8", "train_prop=0.7", "train_prop=0.6"]
            columns = pd.MultiIndex.from_product(
                [possible_sub_experiments, train_props, ["mean", "std"]],
                names=["sub_experiment", "train_prop", "stat"]
            )
        
        table_primary_metric = pd.DataFrame(index=self.datasets, columns=columns)
        table_secondary_metric = pd.DataFrame(index=self.datasets, columns=columns)

        # Insert a column for information on metrics
        table_primary_metric.insert(0, "metric", pd.NA)
        table_secondary_metric.insert(0, "metric", pd.NA)

        if model == "GraphCL_1":
            model_dir_name = "3DInfomax_GraphCL"
        elif model == "GraphCL_2":
            model_dir_name = "GraphMVP_GraphCL"
        else:
            model_dir_name = model

        for dataset in self.datasets:
            path_to_sub_experiments = os.path.join("Results", experiment, model_dir_name, dataset)
            if experiment == "noise":
                sub_experiments = os.listdir(path_to_sub_experiments)
            else:
                sub_experiments = []
                for strat in os.listdir(path_to_sub_experiments):
                    strat_path = os.path.join(path_to_sub_experiments, strat)
                    for subdir in os.listdir(strat_path):
                        full_path = os.path.join(strat, subdir)  # e.g., "random/train_prop=0.6"
                        sub_experiments.append(full_path)
            
            task_type = self.get_dataset_task_type(dataset)
            primary_metric, secondary_metric = self.get_task_type_metrics(task_type)

            for sub_experiment in sub_experiments:
                path_to_seeds = os.path.join(path_to_sub_experiments, sub_experiment)
                seeds = os.listdir(path_to_seeds)
                
                # compute mean and std for each sub-experiment
                results = {}
                eval_file_exists = False
                for seed in seeds:
                    eval_path = os.path.join(path_to_seeds, seed, "evaluation.txt")

                    if not os.path.exists(eval_path):
                        continue  # Skip this seed if file is missing

                    eval_file_exists = True
                    with open(eval_path, "r") as file:
                        for line in file:
                            key, value = line.strip().split(": ")
                            value = float(value)
                            if key not in results:
                                results[key] = []
                            results[key].append(value)
                
                if not eval_file_exists:
                    continue # Skip computations if no evaluation file exists
                
                # if primary_metric == 'rmse':
                #     k = 10   # keep k best runs 
                #     runs_to_keep = np.argsort(np.argsort(results[f'test_{primary_metric}'])) < k

                #     from itertools import compress
                #     for key in results.keys():
                #         results[key] = list(compress(results[key], runs_to_keep))
                
                
        
                if experiment == "split":
                    sub_experiment_key = tuple(os.path.normpath(sub_experiment).split(os.sep)) # In split we have scaff/train_prop=0.8 etc
                else:
                    sub_experiment_key = (sub_experiment,)                 # In perturbation we have noise=0.05 etc
                # Insert values in both tables
                for table, metric in [(table_primary_metric, primary_metric), (table_secondary_metric, secondary_metric)]:
                    if metric in ['rocauc', 'prcauc']:
                        result_mean = np.mean([100*result for result in results[f'{partition}_{metric}']]) 
                        result_std = np.std([100*result for result in results[f'{partition}_{metric}']]) 
                        
                    else:
                        result_mean = np.mean(results[f'{partition}_{metric}'])
                        result_std = np.std(results[f'{partition}_{metric}'])

                    table.loc[dataset, (*sub_experiment_key, "mean")] = result_mean
                    table.loc[dataset, (*sub_experiment_key, "std")] = result_std
                    table.loc[dataset, "metric"] = metric
        
        return table_primary_metric, table_secondary_metric