import MetricDifferenceTableGenerator as mdtg
import LossDifferenceTableGenerator as ldtg
import RawTableGenerator as tg
if __name__ == "__main__":
    # Create an instance of the DeltaTableGenerator class
    loss_diff_table_generator = ldtg.LossDifferenceTableGenerator(
        model="3DInfomax",
        experiment="split",
        decimals=3
    )
    loss_diff_table_generator.print_result_table()

    # raw_table_generator = tg.RawTableGenerator(
    #     model="GraphMVP",
    #     experiment="split",
    #     partition="train",
    #     decimals=3
    # )
    # raw_table_generator.print_result_table(print_secondary_metric=True)

