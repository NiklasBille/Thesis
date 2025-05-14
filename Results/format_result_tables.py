import os
import sys
import argparse

import numpy as np
import pandas as pd

def get_dataset_task_type(dataset):
    dataset_task_types = {
        'esol': 'regression',
        'freesolv': 'regression',
        'lipo': 'regression',
        'hiv': 'classification',
        'bace': 'classification',
        'bbbp': 'classification',
        'tox21': 'classification',
        'toxcast': 'classification',
        'sider': 'classification',
        'clintox': 'classification'
    }
    
    return dataset_task_types[dataset]

def get_task_type_metrics(task_type):
    primary_metric = "rmse" if task_type == "regression" else "rocauc"
    secondary_metric = "mae" if task_type == "regression" else "prcauc"

    return primary_metric, secondary_metric


def create_table(datasets, experiment=None, model=None, partition=None):
    allowed_experiments = ["noise", "split"]
    if experiment not in allowed_experiments:
        raise ValueError(f"Invalid experiment '{experiment}'. Must be one of {allowed_experiments}")
    
    allowed_models = ["3DInfomax", "GraphMVP", "GraphCL"]
    if model not in allowed_models:
        raise ValueError(f"Invalid model '{model}'. Must be one of {allowed_models}")
    
    allowed_partitions = ["train", "val", "test"]
    if partition not in allowed_partitions:
        raise ValueError(f"Invalid partition '{partition}'. Must be one of {allowed_partitions}")
    
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
    
    table_primary_metric = pd.DataFrame(index=datasets, columns=columns)
    table_secondary_metric = pd.DataFrame(index=datasets, columns=columns)

    # Insert a column for information on metrics
    table_primary_metric.insert(0, "metric", pd.NA)
    table_secondary_metric.insert(0, "metric", pd.NA)

    for dataset in datasets:
        path_to_sub_experiments = os.path.join("Results", experiment, model, dataset)
        if experiment == "noise":
            sub_experiments = os.listdir(path_to_sub_experiments)
        else:
            sub_experiments = []
            for strat in os.listdir(path_to_sub_experiments):
                strat_path = os.path.join(path_to_sub_experiments, strat)
                for subdir in os.listdir(strat_path):
                    full_path = os.path.join(strat, subdir)  # e.g., "random/train_prop=0.6"
                    sub_experiments.append(full_path)
        
        task_type = get_dataset_task_type(dataset)
        primary_metric, secondary_metric = get_task_type_metrics(task_type)

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

            if experiment == "split":
                sub_experiment_key = tuple(sub_experiment.split("\\")) # In split we have scaff/train_prop=0.8 etc
            else:
                sub_experiment_key = (sub_experiment,)                 # In perturbation we have noise=0.05 etc
            # Insert values in both tables
            for table, metric in [(table_primary_metric, primary_metric), (table_secondary_metric, secondary_metric)]:
                table.loc[dataset, (*sub_experiment_key, "mean")] = np.mean(results[f'{partition}_{metric}'])
                table.loc[dataset, (*sub_experiment_key, "std")] = np.std(results[f'{partition}_{metric}'])
                table.loc[dataset, "metric"] = metric



    
    return table_primary_metric, table_secondary_metric

def print_experiment_metadata(args):
    print("\n" + "="*80)
    print(f"MODEL: {args.model} | EXPERIMENT: {args.experiment} | PARITION: {args.partition}")
    print("="*80)

def print_result_table(result_df, title):
    print(f"\n {title}")
    print(result_df.to_string())
    print("\n" + "-"*80)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate metric tables for experiments.")
    parser.add_argument('--model', required=True, choices=["3DInfomax", "GraphMVP", "GraphCL"], help="Model name")
    parser.add_argument('--experiment', required=True, choices=["noise", "split"], help="Experiment type")
    parser.add_argument('--partition', required=True, choices=["train", "val", "test"], help="Data partition")
    parser.add_argument('--print_decimals', default=3, type=int, help="How many decimals to print")
    parser.add_argument('--print_secondary_metric', action='store_true', help="Whether the table with secondary metrics is part of output")
    args = parser.parse_args()

    datasets = ["freesolv", "esol", "lipo", "bace", "bbbp", "clintox", "hiv", "sider", "toxcast", "tox21"]
    table_primary_metric, table_secondary_metric = create_table(datasets, experiment=args.experiment, model=args.model, partition=args.partition)
    
    for df in [table_primary_metric, table_secondary_metric]:
        # Get all columns except 'metric'
        value_columns = df.columns.drop("metric", level=0)
        
        # Convert those columns to float and round them
        df[value_columns] = (
            df[value_columns]
            .apply(pd.to_numeric, errors="coerce")  # safely convert to float
            .round(decimals=args.print_decimals)  # round to 2 decimals
        )

    print_experiment_metadata(args)
    print_result_table(table_primary_metric, title="PRIMARY METRIC TABLE")
    if args.print_secondary_metric:
        print_result_table(table_secondary_metric, title="SECONDARY METRIC TABLE")
     
    