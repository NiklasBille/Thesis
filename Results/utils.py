import sys
import RawTableGenerator as tg 
import MetricDifferenceTableGenerator as mdtg
import SplitInterStratTableGenerator as InterStratTG
from typing import Literal

def extract_tables(list_of_models, experiment, partition, decimals, use_percentage=False,  mode=Literal["raw", "metric_difference"]):
    
    if mode == "raw":
        TableGenerator = tg.RawTableGenerator
        table_kwargs = {
            "experiment": experiment,
            "partition": partition,
            "decimals": decimals,
        }
    elif mode == "metric_difference":
        TableGenerator = mdtg.MetricDifferenceTableGenerator
        table_kwargs = {
            "experiment": experiment,
            "partition": partition,
            "decimals": decimals,
            "use_percentage": use_percentage,
        }
    elif mode == "inter_strat":
        TableGenerator = InterStratTG.SplitInterStratTableGenerator
        table_kwargs = {
            "experiment": experiment,
            "partition": partition,
            "decimals": decimals,
        }
    
    else:
        raise ValueError("Mode bust be 'raw', 'metric_difference' or 'inter_strat'.")

    # Extract tables for all models specified and put them in a dictionary
    table_dict = dict.fromkeys(list_of_models)
    for model in list_of_models:
        table_generator = TableGenerator(model=model, **table_kwargs)
        if mode == "inter_strat":
            primary_table, secondary_table =  table_generator.create_table()
        else:
            primary_table, secondary_table =  table_generator.create_table(experiment=experiment, model=model, partition=partition)
        table_dict[model] = {'primary': primary_table, 'secondary': secondary_table}

    return table_dict