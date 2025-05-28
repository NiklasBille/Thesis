import sys
from typing_extensions import override #to explicitly state when overriding method
import pandas as pd

import RawTableGenerator as tg


class SplitInterStratTableGenerator(tg.RawTableGenerator):
    def __init__(self, model, experiment, partition, decimals=None):
        super().__init__(model, experiment, partition, decimals)
        self.raw_primary_table, self.raw_secondary_table = super().create_table(experiment, model, partition)

    def _compute_delta_table(self, df):
        
        # Create empty table to hold delta values
        columns = ["metric", "train_prop=0.8", "train_prop=0.7", "train_prop=0.6"]
        delta_table = pd.DataFrame(index=self.datasets, columns=columns)

        for index, row in df.iterrows():
            R_random = row.loc['random']
            R_scaffold = row.loc['scaff']

            # Remove std from the table
            R_scaffold = R_scaffold.drop(index=R_scaffold.index[R_scaffold.index.get_level_values('stat') == 'std'])
            R_random = R_random.drop(index=R_random.index[R_random.index.get_level_values('stat') == 'std'])

            # Remove the 'stat' level, but keep the corresponding values
            R_scaffold = R_scaffold.droplevel('stat')
            R_random = R_random.droplevel('stat')
            
            delta = (R_scaffold - R_random)
            delta_table.loc[index] = delta

        delta_table['metric'] = df['metric']     

        return delta_table


    @override
    def create_table(self):
        primary_table = self._compute_delta_table(self.raw_primary_table)
        secondary_table = self._compute_delta_table(self.raw_secondary_table)

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

        