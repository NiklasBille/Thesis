# On random train/valid/test splits

Since Uni-Mol does not come with random splitting we have created a script `create_random_splits.py` that does this. The random split versions of the datasets are stored in `molecular_property_prediction_random` and can be used by using `data_path="/workspace/unimol/data/molecular_property_prediction_random"` in the usual fine-tuning script.

Currently only 80/10/10 splits are implemented.