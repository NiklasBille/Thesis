import DeltaTableGenerator as dtg

if __name__ == "__main__":
    # Create an instance of the DeltaTableGenerator class
    delta_table_generator = dtg.DeltaTableGenerator(
        model="3DInfomax",
        experiment="noise",
        partition="test",
        decimals=3,
        secondary_metric=True
    )

