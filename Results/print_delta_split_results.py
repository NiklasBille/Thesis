import argparse
import DeltaSplitTableGenerator as dstg

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate metric tables for experiments.")
    parser.add_argument('--model', required=True, choices=["3DInfomax", "GraphMVP", "GraphCL"], help="Model name")
    parser.add_argument('--partition', default='test', choices=["train", "val", "test"], help="Data partition")
    parser.add_argument('--print_decimals', default=3, type=int, help="How many decimals to print")
    parser.add_argument('--print_secondary_metric', action='store_true', help="Whether the table with secondary metrics is part of output")
    args = parser.parse_args()


    # Create an instance of the DeltaTableGenerator class
    delta_split_table_generator = dstg.DeltaSplitTableGenerator(
        model=args.model,
        experiment='split',
        partition=args.partition,
        decimals=args.print_decimals
    )

    delta_split_table_generator.print_result_table(print_secondary_metric=args.print_secondary_metric)

