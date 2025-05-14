import RawTableGenerator as tg

class DeltaTableGenerator(tg.RawTableGenerator):
    def __init__(self, model, experiment, partition, decimals=None, secondary_metric=None):
        super().__init__(model, experiment, partition, decimals, secondary_metric)
        print(self.datasets)

        def process_table(self, table):
            pass
            
            

        