import argparse
import sys
import RawTableGenerator as tg 
import ModelComparisonTableGenerator as mctg
from utils import extract_tables


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate metric tables for experiments.")
    parser.add_argument('--model', required=True, nargs='+', choices=["3DInfomax", "GraphMVP", "GraphCL_1", "GraphCL_2", "all"], help="Model name. If set to all, all models are compared")
    parser.add_argument('--experiment', required=True, choices=["noise", "split"], help="Experiment type")
    parser.add_argument('--partition', default='test', choices=["train", "val", "test"], help="Data partition")
    parser.add_argument('--print_decimals', default=3, type=int, help="How many decimals to print")
    parser.add_argument('--print_secondary_metric', action='store_true', help="Whether the table with secondary metrics is part of output")
    parser.add_argument('--to_latex', action='store_true', help="Whether to print the table as LaTeX")
    args = parser.parse_args()

    if len(args.model) == 1 and args.model[0] !="all":
        model = args.model[0]
        table_generator = tg.RawTableGenerator(model=model, experiment=args.experiment, partition=args.partition, decimals=args.print_decimals)

    else:
        if args.model == ['all']:
            models = ["3DInfomax", "GraphCL_1", "GraphMVP", "GraphCL_2"]
        else:
            models = args.model

        table_generator = mctg.ModelComparisonTableGenerator(list_of_models=models, experiment=args.experiment,partition=args.partition,decimals=args.print_decimals)
        table_dict = extract_tables(
            list_of_models=models, experiment=args.experiment, 
            partition=args.partition, decimals=args.print_decimals, 
            mode="raw")
        
        table_generator.set_table_dict(table_dict)

    table_generator.print_result_table(print_secondary_metric=args.print_secondary_metric)
    
    if args.to_latex is True:
        table_generator.convert_table_to_latex(print_secondary_metric=args.print_secondary_metric)
    
