import sys
import RawTableGenerator as tg
import pandas as pd

from typing import Literal
from typing_extensions import override #to explicitly state when overriding metho

class ModelComparisonTableGenerator(tg.RawTableGenerator):
    def __init__(self, experiment, partition, decimals=None):
        super().__init__(experiment=experiment, partition=partition, decimals=decimals, isComparingModels=True)

        self.raw_table_dict = dict.fromkeys(self.allowed_models)
        for model in self.allowed_models:
            raw_primary_table, raw_secondary_table =  super().create_table(experiment, model, partition)
            self.raw_table_dict[model] = {'primary': raw_primary_table, 'secondary': raw_secondary_table}
        
    def _compute_model_comparison_table(self, raw_table_dict, metric_type: Literal['primary', 'secondary']):

        # Create empty MultiIndex table
        if self.experiment == "noise":
            possible_sub_experiments = ["noise=0.0", "noise=0.05", "noise=0.1", "noise=0.2"] 
            columns = pd.MultiIndex.from_product(
                [self.allowed_models, possible_sub_experiments],
                names=["model", "sub_experiment"]
                )
        table = pd.DataFrame(index=self.datasets, columns=columns)

        # Insert a column for information on metrics
        table.insert(0, 'metric', raw_table_dict['3DInfomax'][metric_type]['metric'])

        # Fill table
        for model, raw_tables in raw_table_dict.items():
            raw_table = raw_tables[metric_type]
            
            # Add each sub experiment metrics to the correct colunm
            for sub_experiment in possible_sub_experiments:
                R = raw_table.loc[self.datasets, sub_experiment]
                R = R.drop('std', axis=1)      # Remove 'std' column  
                R.columns.name = None          # Remove index name 'stat'
                table.loc[:, (model, sub_experiment)] = R['mean']

        return table

        

    @override
    def create_table(self):
        primary_table = self._compute_model_comparison_table(self.raw_table_dict, 'primary')
        secondary_table = self._compute_model_comparison_table(self.raw_table_dict, 'secondary')

        return primary_table, secondary_table
    
    @override
    def print_result_table(self, print_secondary_metric=False):
        print("\n" + "="*80)
        print(f"EXPERIMENT: {self.experiment} | PARITION: {self.partition} | Comparing models")
        print("="*80)
        primary_table, secondary_table = self.create_table()
        if self.decimals is not None:
            self.round_table(primary_table)
            self.round_table(secondary_table)
        print("\n PRIMARY METRIC")
        print(primary_table.to_string())
        print("\n" + "-"*80)
        if print_secondary_metric:
            print("\n SECONDARY METRIC")
            print(secondary_table.to_string())
            print("\n" + "-"*80)