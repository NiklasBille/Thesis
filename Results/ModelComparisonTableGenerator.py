import sys
import RawTableGenerator as tg
import pandas as pd

from typing import Literal
from typing_extensions import override #to explicitly state when overriding metho

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
                [possible_sub_experiments, self.list_of_models],
                names=["sub_experiment", "model"]
                )
        else:
            possible_sub_experiments = ["random", "scaff"]
            train_props = ["train_prop=0.8", "train_prop=0.7", "train_prop=0.6"]
            columns = pd.MultiIndex.from_product(
                [possible_sub_experiments, train_props, self.list_of_models],
                names=["sub_experiment", "train_prop", "model"]
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
                    table.loc[:, (sub_experiment, model)] = R['mean']
            
            elif self.experiment =='split':

                for sub_experiment in possible_sub_experiments:
                    for train_prop in train_props:
                        R = raw_table.loc[self.datasets, (sub_experiment, train_prop)]
                        table.loc[:, (sub_experiment, train_prop, model)] = R['mean']

        return table

        

    @override
    def create_table(self):
        primary_table = self._compute_model_comparison_table(self.raw_table_dict, 'primary')
        secondary_table = self._compute_model_comparison_table(self.raw_table_dict, 'secondary')

        return primary_table, secondary_table
    
    @override
    def print_result_table(self, print_secondary_metric=False):
        print("\n" + "="*80)
        print(f"EXPERIMENT: {self.experiment} | PARTITION: {self.partition} | Comparing models")
        print("="*80)
        primary_table, secondary_table = self.create_table()
        if self.decimals is not None:
            self.round_table(primary_table)
            self.round_table(secondary_table)
        print("PRIMARY METRIC")
    
        if self.experiment == 'noise':
            print(primary_table.to_string())
            print("\n" + "-"*80)
            if print_secondary_metric:
                print("\n SECONDARY METRIC")
                print(secondary_table.to_string())
                print("\n" + "-"*80)
            
        # Split table is very large so we present it as two tables, one for random and one for scaffold
        else: 
            print("[RANDOM]")
            primary_table.set_index('metric', append=True, inplace=True) # keep metric column when slicing
            print(primary_table.loc[:, 'random'].to_string(), '\n')
            print("[SCAFFOLD]")
            print(primary_table.loc[:, 'scaff'].to_string(), '\n', '-'*80)

            if print_secondary_metric:
                secondary_table.set_index('metric', append=True, inplace=True) # keep metric column when slicing
                print("SECONDARY METRIC")
                print("[RANDOM]")
                print(secondary_table.loc[:, 'random'].to_string(), '\n')
                print("[SCAFFOLD]")
                print(secondary_table.loc[:, 'scaff'].to_string(), '\n', '-'*80)
