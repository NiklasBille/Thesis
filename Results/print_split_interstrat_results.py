import argparse
import SplitInterStratTableGenerator as InterStratTG

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate metric tables for experiments.")
    parser.add_argument('--model', required=True, choices=["3DInfomax", "GraphMVP", "GraphCL_1", "GraphCL_2"], help="Model name.")
    parser.add_argument('--partition', default='test', choices=["train", "val", "test"], help="Data partition")
    parser.add_argument('--print_decimals', default=3, type=int, help="How many decimals to print")
    parser.add_argument('--print_secondary_metric', action='store_true', help="Whether the table with secondary metrics is part of output")
    parser.add_argument('--to_latex', action='store_true', help="Whether to print the table as LaTeX")
    args = parser.parse_args()

    # Create an instance of the SplitInterStratTableGenerator class
    inter_strat_table_generator = InterStratTG.SplitInterStratTableGenerator(
        model=args.model,
        experiment='split',
        partition=args.partition,
        decimals=args.print_decimals
    )

    inter_strat_table_generator.print_result_table(print_secondary_metric=args.print_secondary_metric)

    if args.to_latex is True:
        inter_strat_table_generator.convert_table_to_latex(print_secondary_metric=args.print_secondary_metric)

