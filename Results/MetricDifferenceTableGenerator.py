import sys
import RawTableGenerator as tg

from typing_extensions import override #to explicitly state when overriding metho

class MetricDifferenceTableGenerator(tg.RawTableGenerator):
    def __init__(self, model, experiment, partition, decimals=None, use_percentage=False):
        super().__init__(model, experiment, partition, decimals)
        self.use_percentage = use_percentage
        self.raw_primary_table, self.raw_secondary_table =  super().create_table(experiment, model, partition)

    def _compute_metric_diff_table(self, df):
        metric_diff_table = df.copy(deep=True)
        # For each dataset
        for index, row in metric_diff_table.iterrows():
            # The table for each experiment is different, so they are processed in different ways
            if self.experiment == 'noise':
                noise_levels = row.index.get_level_values(0).unique()
                noise_levels = noise_levels[(noise_levels != 'metric') & (noise_levels != 'noise=0.0')] 
            
                baseline = row.loc['noise=0.0', 'mean']
                for noise_level in noise_levels:
                    readout = row.loc[noise_level, 'mean']
                    if self.use_percentage is True:
                        metric_diff_table.loc[index, (noise_level, 'mean')] = 100*(readout - baseline)/baseline
                    else:
                        metric_diff_table.loc[index, (noise_level, 'mean')] = readout - baseline

            elif self.experiment =='split':
                train_props = row.index.get_level_values(1).unique()
                train_props = train_props[(train_props != '') & (train_props != 'train_prop=0.8')]

                strategies = row.index.get_level_values(0).unique()   #random, scaff
                strategies = strategies[strategies != 'metric']
                
                for strategy in strategies:
                    baseline = row.loc[strategy, 'train_prop=0.8', 'mean']
                    for train_prop in train_props:
                        readout = row.loc[strategy, train_prop, 'mean']
                        if self.use_percentage is True:
                            metric_diff_table.loc[index, (strategy, train_prop, 'mean')] = 100*(readout - baseline)/baseline
                        else:
                            metric_diff_table.loc[index, (strategy, train_prop, 'mean')] = readout - baseline

        return metric_diff_table
    
    def set_use_percentage(self, use_percentage):
        self.use_percentage = use_percentage

    @override
    def create_table(self, experiment, model, partition):
        primary_table = self._compute_metric_diff_table(self.raw_primary_table)
        secondary_table = self._compute_metric_diff_table(self.raw_secondary_table)

        return primary_table, secondary_table


    @override
    def print_result_table(self, print_secondary_metric=False):
        print("\n" + "="*80)
        print(f"MODEL: {self.model} | EXPERIMENT: {self.experiment} | PARTITION: {self.partition} | FORMAT: Relative changes")
        print("="*80)
        primary_table, secondary_table = self.create_table(self.experiment, self.model, self.partition)
        if self.decimals is not None:
            self.round_table(primary_table)
            self.round_table(secondary_table)

        if print_secondary_metric is False:
            print("PRIMARY METRIC TABLE")
            print(primary_table.to_string())
            print("\n" + "-"*80)
        else:
            print("SECONDARY METRIC TABLE")
            print(secondary_table.to_string())
            print("\n" + "-"*80)



    
            
            

        