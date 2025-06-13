import argparse
import sys
import RawTableGenerator as tg 
import pandas as pd
import numpy as np

from utils import extract_tables
import MetricDifferenceTableGenerator as mdtg
import ModelComparisonTableGenerator as mctg

def extract_difference_table(df1, df2, experiment):
    if experiment == "noise":
        mean_df1 = df1.loc[:, 
                    (df1.columns.get_level_values(0) != 'noise=0.0') &
                    (df1.columns.get_level_values(2) == 'mean')
                    ]
        
        mean_df2 = df2.loc[:, 
                    (df2.columns.get_level_values(0) != 'noise=0.0') &
                        (df2.columns.get_level_values(2) == 'mean')
                    ]
        
        # Compute 3DModel - 2DModel for each noise level
        model_gap_df1 = mean_df1.groupby(level=0, axis=1).apply(lambda x: x.iloc[:, 0] - x.iloc[:, 1])
        model_gap_df2 = mean_df2.groupby(level=0, axis=1).apply(lambda x: x.iloc[:, 0] - x.iloc[:, 1])

        difference_table = abs(model_gap_df1.sub(model_gap_df2))

    elif experiment == "split":
        
        mean_df1 = df1.loc[:, 
            (df1.columns.get_level_values(1) != 'train_prop=0.8') &
            (df1.columns.get_level_values(3) == 'mean')
            ]
        
        mean_df2 = df2.loc[:, 
            (df2.columns.get_level_values(1) != 'train_prop=0.8') &
            (df2.columns.get_level_values(3) == 'mean')
            ]
        
        # Compute 3DModel - 2DModel for each strategy and its split levels
        model_gap_df1 = mean_df1.groupby(level=[0, 1], axis=1).apply(lambda x: x.iloc[:, 0] - x.iloc[:, 1])
        model_gap_df2 = mean_df2.groupby(level=[0, 1], axis=1).apply(lambda x: x.iloc[:, 0] - x.iloc[:, 1])

        # Ensure train_prop=0.7 is the first column
        model_gap_df1 = model_gap_df1.reindex(columns=['train_prop=0.7', 'train_prop=0.6'], level=1)
        model_gap_df2 = model_gap_df2.reindex(columns=['train_prop=0.7', 'train_prop=0.6'], level=1)

        difference_table = abs(model_gap_df1.sub(model_gap_df2))
        difference_table = difference_table.reindex(columns=['scaff', 'random'], level=0)
        

    
    

            
    return difference_table


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate metric tables for experiments.")
    parser.add_argument('--model', required=True, choices=["3DInfomax", "GraphMVP"], help="Model name")
    parser.add_argument('--experiment', required=True, choices=["noise", "split"], help="Experiment type")
    parser.add_argument('--partition', default='test', choices=["train", "val", "test"], help="Data partition")
    parser.add_argument('--print_decimals', default=3, type=int, help="How many decimals to print")
    args = parser.parse_args()

    #table_generator = tg.RawTableGenerator(model=args.model, experiment=args.experiment, partition=args.partition, decimals=args.print_decimals)
    if args.model == "3DInfomax":
        models_to_compare = [args.model, "GraphCL_1"]
    elif args.model == "GraphMVP":
        models_to_compare = [args.model, "GraphCL_2"]

    table_generator = mctg.ModelComparisonTableGenerator(list_of_models=models_to_compare, experiment=args.experiment,partition=args.partition,decimals=args.print_decimals)
    table_dict = extract_tables(
            list_of_models=models_to_compare, experiment=args.experiment, 
            partition=args.partition, decimals=args.print_decimals, 
            mode="metric_difference", use_percentage=True)

    table_generator.set_table_dict(table_dict)
    primary_table = table_generator.get_table(secondary_metric=False)
    secondary_table = table_generator.get_table(secondary_metric=True)

    difference_table = extract_difference_table(df1=primary_table, df2=secondary_table, experiment=args.experiment)
    
    difference_table = (
        difference_table
        .apply(pd.to_numeric, errors="coerce")  # safely convert to float
        .round(decimals=2)
        )
    
    print(difference_table)