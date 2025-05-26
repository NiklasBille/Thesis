import os
import sys
import yaml

# Custom Dumper that capitalizes booleans
class CustomDumper(yaml.SafeDumper):
    pass

def bool_representer(dumper, data):
    return dumper.represent_scalar('tag:yaml.org,2002:bool', 'True' if data else 'False')

class FlowList(list): pass

def flow_list_representer(dumper, data):
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

CustomDumper.add_representer(bool, bool_representer)
CustomDumper.add_representer(FlowList, flow_list_representer)

# Define datasets and noise levels
datasets = ['esol', 'freesolv', 'lipophilicity', 'hiv', 'bace', 'bbbp', 'tox21', 'toxcast', 'sider', 'clintox', 'muv']
split_types = ['scaff', 'random']
train_props = [0.8, 0.7, 0.6]

dataset_task_type = {
    'esol': 'regression',
    'freesolv': 'regression',
    'lipophilicity': 'regression',
    'hiv': 'classification',
    'bace': 'classification',
    'bbbp': 'classification',
    'tox21': 'classification',
    'toxcast': 'classification',
    'sider': 'classification',
    'clintox': 'classification',
    'muv': 'classification',
}

# Base config structure
def create_config(dataset, train_prop, split_type):
    split = 'scaffold' if split_type == 'scaff' else 'random'
    dataset_naming = 'lipo' if dataset == 'lipophilicity' else dataset
    return {
        'output_model_dir': f'../runs/split/GraphCL/{dataset_naming}/{split_type}/train_prop={train_prop}', 
        'input_model_file': f'../weights/pretrained/GraphCL_{dataset_task_type[dataset]}.pth',
        'multiple_seeds': [4, 5, 6], 
        'train_prop': train_prop, 
        'split': split,
        'dataset': dataset,
        'epochs': 1000,
        'patience': 35,
        'minimum_epochs': 120,
    }

# Output base dir
base_dir = "configs_split_experiments/GraphCL"
os.makedirs(base_dir, exist_ok=True)

# Generate config files
for dataset in datasets:
    dataset_naming = 'lipo' if dataset == 'lipophilicity' else dataset
    dataset_dir = os.path.join(base_dir, dataset_naming)
    os.makedirs(dataset_dir, exist_ok=True)
    for split_type in split_types:
        split_dir = os.path.join(dataset_dir, split_type) 
        os.makedirs(split_dir, exist_ok=True)             
        for train_prop in train_props:
            config = create_config(dataset, train_prop, split_type)
            filename = os.path.join(dataset_dir, f"{split_type}", f"train_prop={train_prop}.yml")
            with open(filename, 'w') as f:
                yaml.dump(config, f, Dumper=CustomDumper, sort_keys=False, default_flow_style=False)
            print(f"Created: {filename}")
            
