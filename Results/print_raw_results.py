import argparse
import TableGenerator as tg 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate metric tables for experiments.")
    parser.add_argument('--model', required=True, choices=["3DInfomax", "GraphMVP", "GraphCL"], help="Model name")
    parser.add_argument('--experiment', required=True, choices=["noise", "split"], help="Experiment type")
    parser.add_argument('--partition', required=True, choices=["train", "val", "test"], help="Data partition")
    parser.add_argument('--print_decimals', default=3, type=int, help="How many decimals to print")
    parser.add_argument('--print_secondary_metric', action='store_true', help="Whether the table with secondary metrics is part of output")
    args = parser.parse_args()


    # Create an instance of the TableGenerator class
    table_generator = tg.TableGenerator(
        model=args.model,
        experiment=args.experiment,
        parition=args.partition,
        decimals=args.print_decimals,
        secondary_metric=args.print_secondary_metric
    )
    table_generator.print_result_table(print_secondary_metric=args.print_secondary_metric)
    
