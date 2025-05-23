import argparse
import SplitIntraStratTableGenerator as IntraStratTG

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate metric tables for experiments.")
    parser.add_argument('--partition', default='test', choices=["train", "val", "test"], help="Data partition")
    parser.add_argument('--print_decimals', default=3, type=int, help="How many decimals to print")
    parser.add_argument('--print_secondary_metric', action='store_true', help="Whether the table with secondary metrics is part of output")
    args = parser.parse_args()


    # Create an instance of the DeltaTableGenerator class
    intra_strat_table_generator = IntraStratTG.SplitIntraStratTableGenerator(
        experiment='split',
        partition=args.partition,
        decimals=args.print_decimals
    )

    intra_strat_table_generator.print_result_table(print_secondary_metric=args.print_secondary_metric)

