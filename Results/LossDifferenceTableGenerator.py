import RawTableGenerator as tg
import pandas as pd
import numpy as np
import os

class LossDifferenceTableGenerator(tg.RawTableGenerator):
    def __init__(self, model, experiment, decimals=None):
        super().__init__(model=model, experiment=experiment, decimals=decimals, secondary_metric=False, partition=None)
        
    def print_result_table(self):
        print("\n" + "="*80)
        print(f"MODEL: {self.model} | EXPERIMENT: {self.experiment}")
        print("="*80)
        loss_difference_metric = self.create_table()
        if self.decimals is not None:
            loss_difference_metric = loss_difference_metric.apply(pd.to_numeric, errors="coerce").round(decimals=self.decimals)  
        print("\n LOSS DIFFERENCE TABLE")
        print(loss_difference_metric.to_string())
        print("\n" + "-"*80)
    


    def create_table(self):     
        # Create empty MultiIndex table
        if self.experiment == "noise":
            possible_sub_experiments = ["noise=0.0", "noise=0.05", "noise=0.1", "noise=0.2"] 
            columns = pd.MultiIndex.from_product(
                [possible_sub_experiments, ["mean", "std"]],
                names=["sub_experiment", "stat"]
                )
        if self.experiment == "split":
            possible_sub_experiments = ["random", "scaff"]
            train_props = ["train_prop=0.8", "train_prop=0.7", "train_prop=0.6"]
            columns = pd.MultiIndex.from_product(
                [possible_sub_experiments, train_props, ["mean", "std"]],
                names=["sub_experiment", "train_prop", "stat"]
            )
        
        loss_difference_metric = pd.DataFrame(index=self.datasets, columns=columns)

        # Insert a column for information on metrics
        # loss_difference_metric.insert(0, "metric", pd.NA)

        for dataset in self.datasets:
            path_to_sub_experiments = os.path.join("Results", self.experiment, self.model, dataset)
            if self.experiment == "noise":
                sub_experiments = os.listdir(path_to_sub_experiments)
            else:
                sub_experiments = []
                for strat in os.listdir(path_to_sub_experiments):
                    strat_path = os.path.join(path_to_sub_experiments, strat)
                    for subdir in os.listdir(strat_path):
                        full_path = os.path.join(strat, subdir)  # e.g., "random/train_prop=0.6"
                        sub_experiments.append(full_path)

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

                if self.experiment == "split":
                    sub_experiment_key = tuple(os.path.normpath(sub_experiment).split(os.sep)) # In split we have scaff/train_prop=0.8 etc
                else:
                    sub_experiment_key = (sub_experiment,)                 # In perturbation we have noise=0.05 etc
            
                required_keys = ['test_loss', 'val_loss']
                if all(key in results for key in required_keys):
                    test_losses = np.array(results['test_loss'])
                    val_losses = np.array(results['val_loss'])
                    diff = np.abs(test_losses - val_losses)
                    loss_difference_metric.loc[dataset, (*sub_experiment_key, "mean")] = np.mean(diff)
                    loss_difference_metric.loc[dataset, (*sub_experiment_key, "std")] = np.std(diff)
                    # loss_difference_metric.loc[dataset, "metric"] = "|test_loss - val_loss|"
                else:
                    print(f"Missing keys in results for dataset={dataset}, sub_experiment={sub_experiment}")

        
        return loss_difference_metric
            
            

        