import sys
from typing_extensions import override #to explicitly state when overriding method
from typing import Literal
import pandas as pd

import RawTableGenerator as tg


class SplitIntraStratTableGenerator(tg.RawTableGenerator):
    def __init__(self, experiment, partition, decimals=None):
        super().__init__(experiment=experiment, partition=partition, decimals=decimals, isComparingModels=True)
        
        # TODO implement for GraphMVP and GraphCL_2 
        self.allowed_models = ['GraphCL_1', '3DInfomax']
        self.raw_table_dict = dict.fromkeys(self.allowed_models)
        for model in self.allowed_models:
            raw_primary_table, raw_secondary_table =  super().create_table(experiment, model, partition)
            self.raw_table_dict[model] = {'primary': raw_primary_table, 'secondary': raw_secondary_table}

    def _compute_delta_table(self, raw_table_dict, metric_type: Literal['primary', 'secondary']):
        possible_sub_experiments = ["random", "scaff"]
        train_props = ["train_prop=0.8", "train_prop=0.7", "train_prop=0.6"]
        columns = pd.MultiIndex.from_product(
            [possible_sub_experiments, train_props, self.allowed_models],
            names=["sub_experiment", "train_prop", "model"]
        )

        # Create empty table to hold delta values    
        table = pd.DataFrame(index=self.datasets, columns=columns)

        # Insert a column for information on metrics
        table.insert(0, 'metric', raw_table_dict['3DInfomax'][metric_type]['metric'])
        # Fill table
        for model, raw_tables in raw_table_dict.items():
            raw_table = raw_tables[metric_type]

            # Add each sub experiment metrics to the correct colunm
            for sub_experiment in possible_sub_experiments:
                for train_prop in train_props:
                    R = raw_table.loc[self.datasets, (sub_experiment, train_prop)]
                    
                    # 2D model is always our baseline in this comparison
                    if model == 'GraphCL_1':
                        table.loc[:, (sub_experiment, train_prop, model)] = R['mean']
                    
                    elif model == '3DInfomax':
                        baseline = table.loc[:, (sub_experiment, train_prop, 'GraphCL_1')]
                        table.loc[:, (sub_experiment, train_prop, model)] = R['mean'] - baseline
        
        return table


    @override
    def create_table(self):
        primary_table = self._compute_delta_table(self.raw_table_dict, 'primary')
        secondary_table = self._compute_delta_table(self.raw_table_dict, 'secondary')

        return primary_table, secondary_table

    @override
    def print_result_table(self, print_secondary_metric=False):
        print("\n" + "="*80)
        print(f"EXPERIMENT: {self.experiment} | PARTITION: {self.partition} | Intra-strategy compare")
        print("="*80)
        primary_table, secondary_table = self.create_table()
        if self.decimals is not None:
            self.round_table(primary_table)
            self.round_table(secondary_table)
        print("PRIMARY METRIC DELTA TABLE")
        print(primary_table.to_string())
        print("\n" + "-"*80)
        if print_secondary_metric:
            print("SECONDARY METRIC DELTA TABLE")
            print(secondary_table.to_string())
            print("\n" + "-"*80)

        