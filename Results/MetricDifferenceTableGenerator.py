import sys
import RawTableGenerator as tg

from typing_extensions import override #to explicitly state when overriding metho

class MetricDifferenceTableGenerator(tg.RawTableGenerator):
    def __init__(self, model, experiment, partition, decimals=None):
        super().__init__(model, experiment, partition, decimals)
        self.primary_table, self.secondary_table = self.create_table(experiment, model, partition)


    def process_table(self):
        # Process both primary metrics and secondary metrics
        for df in [self.primary_table, self.secondary_table]:

            # For each dataset
            for index, row in df.iterrows():
                # The table for each experiment is different, so they are processed in different ways

                if self.experiment == 'noise':
                    noise_levels = row.index.get_level_values(0).unique()
                    noise_levels = noise_levels[(noise_levels != 'metric') & (noise_levels != 'noise=0.0')] 
                
                    baseline = row.loc['noise=0.0', 'mean']
                    for noise_level in noise_levels:
                        readout = row.loc[noise_level, 'mean']
                        df.loc[index, (noise_level, 'mean')] = baseline - readout

                elif self.experiment =='split':
                    train_props = row.index.get_level_values(1).unique()
                    train_props = train_props[(train_props != '') & (train_props != 'train_prop=0.8')]

                    strategies = row.index.get_level_values(0).unique()   #random, scaff
                    strategies = strategies[strategies != 'metric']
                    
                    for strategy in strategies:
                        baseline = row.loc[strategy, 'train_prop=0.8', 'mean']
                        for train_prop in train_props:
                            readout = row.loc[strategy, train_prop, 'mean']
                            df.loc[index, (strategy, train_prop, 'mean')] = baseline - readout

                else:
                    pass

    @override
    def print_result_table(self, print_secondary_metric=False):
        print("\n" + "="*80)
        print(f"MODEL: {self.model} | EXPERIMENT: {self.experiment} | PARITION: {self.partition} | FORMAT: Relative changes")
        print("="*80)
        if self.decimals is not None:
            self.round_table(self.primary_table)
            self.round_table(self.secondary_table)

        print("\n PRIMARY METRIC TABLE")
        print(self.primary_table.to_string())
        print("\n" + "-"*80)
        if print_secondary_metric:
            print("\n SECONDARY METRIC TABLE")
            print(self.secondary_table.to_string())
            print("\n" + "-"*80)



    
            
            

        