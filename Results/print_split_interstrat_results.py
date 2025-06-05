import sys
import argparse
import SplitInterStratTableGenerator as InterStratTG
import ModelComparisonTableGenerator as mctg

from utils import extract_tables

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate metric tables for experiments.")
    parser.add_argument('--model', required=True, nargs='+', choices=["3DInfomax", "GraphMVP", "GraphCL_1", "GraphCL_2", "all"], help="Model name. If set to all, all models are compared")
    parser.add_argument('--partition', default='test', choices=["train", "val", "test"], help="Data partition")
    parser.add_argument('--print_decimals', default=3, type=int, help="How many decimals to print")
    parser.add_argument('--print_secondary_metric', action='store_true', help="Whether the table with secondary metrics is part of output")
    parser.add_argument('--to_latex', action='store_true', help="Whether to print the table as LaTeX")
    args = parser.parse_args()
    

    if len(args.model) == 1 and args.model[0] != "all":
        # Create an instance of the SplitInterStratTableGenerator class
        table_generator = InterStratTG.SplitInterStratTableGenerator(model=args.model, experiment='split', partition=args.partition, decimals=args.print_decimals)
    
    else:
        if args.model == ['all']:
            models = ["GraphCL_1", "3DInfomax", "GraphCL_2", "GraphMVP"]
        else:
            models = args.model

        table_generator = mctg.ModelComparisonSplitInterStratTableGenerator(list_of_models=models, experiment="split",partition=args.partition,decimals=args.print_decimals)

        table_dict = extract_tables(
            list_of_models=models, experiment="split", 
            partition=args.partition, decimals=args.print_decimals, 
            mode="inter_strat")
        
        table_generator.set_table_dict(table_dict)
    
    table_generator.print_result_table(print_secondary_metric=args.print_secondary_metric)

    if args.to_latex is True:
        table_generator.convert_table_to_latex(print_secondary_metric=args.print_secondary_metric)

    sys.exit()

    if args.model == "all":
        models = ["GraphCL_1", "3DInfomax", "GraphCL_2", "GraphMVP"]
        table_generator = mctg.ModelComparisonSplitInterStratTableGenerator(list_of_models=models, experiment="split",partition=args.partition,decimals=args.print_decimals)

        table_dict = extract_tables(
            list_of_models=models, experiment="split", 
            partition=args.partition, decimals=args.print_decimals, 
            mode="inter_strat")
        
        table_generator.set_table_dict(table_dict)
    else:
        pass


    table_generator.print_result_table(print_secondary_metric=args.print_secondary_metric)

    if args.to_latex is True:
        table_generator.convert_table_to_latex(print_secondary_metric=args.print_secondary_metric)




