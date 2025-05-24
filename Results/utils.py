def extract_tables(list_of_models, experiment, partition, decimals, TableGenerator):
    # Extract tables for all models specified and put them in a dictionary
    table_dict = dict.fromkeys(list_of_models)
    for model in list_of_models:
        raw_table_generator = TableGenerator(
                model=model, 
                experiment=experiment, 
                partition=partition, 
                decimals=decimals
        )
        primary_table, secondary_table =  raw_table_generator.create_table(experiment, model, partition)
        table_dict[model] = {'primary': primary_table, 'secondary': secondary_table}

    return table_dict