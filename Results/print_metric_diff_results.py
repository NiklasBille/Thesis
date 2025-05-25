import argparse
import sys
from utils import extract_tables

import MetricDifferenceTableGenerator as mdtg
import ModelComparisonTableGenerator as mctg

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate metric tables for experiments.")
    parser.add_argument('--model', required=True, nargs='+', choices=["3DInfomax", "GraphMVP", "GraphCL_1", "GraphCL_2", "all"], help="Model name. If set to all, all models are compared")
    parser.add_argument('--experiment', required=True, choices=["noise", "split"], help="Experiment type")
    parser.add_argument('--partition', default='test', choices=["train", "val", "test"], help="Data partition")
    parser.add_argument('--print_decimals', default=3, type=int, help="How many decimals to print")
    parser.add_argument('--print_secondary_metric', action='store_true', help="Whether the table with secondary metrics is part of output")
    parser.add_argument('--to_latex', action='store_true', help="Whether to print the table as LaTeX")
    parser.add_argument('--percentage', action='store_true', help="Whether the comparison is percentage based")
    args = parser.parse_args()

    if len(args.model) == 1 and args.model[0] !="all":
        model = args.model[0]
        table_generator = mdtg.MetricDifferenceTableGenerator(model=model, experiment=args.experiment, partition=args.partition, decimals=args.print_decimals)
    
    else:
        if args.model == ['all']:
            models = ["3DInfomax", "GraphCL_1", "GraphMVP", "GraphCL_2"]
        else:
            models = args.model

        table_generator = mctg.ModelComparisonTableGenerator(list_of_models=models, experiment=args.experiment,partition=args.partition,decimals=args.print_decimals)
        table_dict = extract_tables(
            list_of_models=models, experiment=args.experiment, 
            partition=args.partition, decimals=args.print_decimals, 
            mode="metric_difference", use_percentage=args.percentage)

        table_generator.set_table_dict(table_dict)

    table_generator.print_result_table(print_secondary_metric=args.print_secondary_metric)

    sys.exit()

    # If only one model or all models are specified
    if len(args.model) == 1:
        args.model = args.model[0]

        if not args.model == "all":
            table_generator = mdtg.MetricDifferenceTableGenerator(
                model=args.model,
                experiment=args.experiment,
                partition=args.partition,
                decimals=args.print_decimals
            )

            # Create an instance of the DeltaTableGenerator class
            table_generator.print_result_table(print_secondary_metric=args.print_secondary_metric)
        
        else:
            table_generator = mctg.ModelComparisonTableGenerator(
                list_of_models=["3DInfomax", "GraphCL_1", "GraphMVP", "GraphCL_2"],
                experiment=args.experiment,
                partition=args.partition,
                decimals=args.print_decimals
            )

            # First we compute delta tables for all models and put them in a dictionary
            allowed_models = table_generator.allowed_models
            table_dict = dict.fromkeys(allowed_models)
            for model in allowed_models:
                delta_table_generator = mdtg.MetricDifferenceTableGenerator(
                        model=model, experiment=args.experiment, partition=args.partition, decimals=args.print_decimals
                        )
                primary_table, secondary_table =  delta_table_generator.create_table(experiment=args.experiment, model=model, partition=args.partition)
                table_dict[model] = {'primary': primary_table, 'secondary': secondary_table}
            
            # Then we inject the new metric table dictionary 
            table_generator.set_table_dict(table_dict)
            table_generator.print_result_table(print_secondary_metric=args.print_secondary_metric)

    else:
        table_generator = mctg.ModelComparisonTableGenerator(
            experiment=args.experiment,
            partition=args.partition,
            list_of_models=args.model,
            decimals=args.print_decimals
        )

        # First we extract tables for all models specified and put them in a dictionary
        table_dict = dict.fromkeys(args.model)
        for model in args.model:
            delta_table_generator = mdtg.MetricDifferenceTableGenerator(
                    model=model, 
                    experiment=args.experiment, 
                    partition=args.partition, 
                    decimals=args.print_decimals,
                    use_percentage=args.percentage
            )
            primary_table, secondary_table =  delta_table_generator.create_table(experiment=args.experiment, model=model, partition=args.partition)
            table_dict[model] = {'primary': primary_table, 'secondary': secondary_table}
        
        # Then we inject the new metric table dictionary 
        table_generator.set_table_dict(table_dict)
        table_generator.print_result_table(print_secondary_metric=args.print_secondary_metric, use_percentage=args.percentage)

    if args.to_latex is True:
        table_generator.convert_table_to_latex(print_secondary_metric=args.print_secondary_metric)

        





