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
datasets = ['esol', 'freesolv', 'lipo']
split_types = ['scaff', 'random']
train_props = [0.8, 0.7, 0.6]

# Base config structure
def create_config(dataset, train_prop, split_type):
    force_random_split = True if split_type == 'random' else False
    return {
        'experiment_name': f'3DInfomax_{dataset}_{split_type}={train_prop}',
        'logdir': f'runs/split/3DInfomax/_test/{dataset}/{split_type}/train_prop={train_prop}', # remove test/ for actual runs
        'multiple_seeds': [1, 2], # for testing, change to [1, 2, 3]
        'train_prop': train_prop, 
        'force_random_split': force_random_split,
        'pretrain_checkpoint': 'runs/PNA_qmugs_NTXentMultiplePositives_620000_123_25-08_09-19-52/best_checkpoint_35epochs.pt',
        'transfer_layers': ['gnn.'],
        'dataset': f'ogbg-mol{dataset}',
        'num_epochs': 2, # Set to 1 for testing, change to 1000 for actual runs
        'batch_size': 30,
        'log_iterations': 30,
        'patience': 40,
        'minimum_epochs': 120,
        'loss_func': 'L1Loss',
        'required_data': ['dgl_graph', 'targets'],
        'metrics': ['mae', 'rmse'],
        'optimizer': 'Adam',
        'optimizer_params': {
            'lr': 1.0e-3
        },
        'scheduler_step_per_batch': False,
        'lr_scheduler': 'WarmUpWrapper',
        'lr_scheduler_params': {
            'warmup_steps': FlowList([700, 700, 350]),
            'interpolation': 'linear',
            'wrapped_scheduler': 'ReduceLROnPlateau',
            'factor': 0.5,
            'patience': 25,
            'min_lr': 1.0e-6,
            'mode': 'min',
            'verbose': True
        },
        'model_type': 'PNA',
        'model_parameters': {
            'target_dim': 1,
            'hidden_dim': 200,
            'mid_batch_norm': True,
            'last_batch_norm': True,
            'readout_batchnorm': True,
            'batch_norm_momentum': 0.1,
            'readout_hidden_dim': 200,
            'readout_layers': 2,
            'dropout': 0.0,
            'propagation_depth': 7,
            'aggregators': ['mean', 'max', 'min', 'std'],
            'scalers': ['identity', 'amplification', 'attenuation'],
            'readout_aggregators': ['min', 'max', 'mean', 'sum'],
            'pretrans_layers': 2,
            'posttrans_layers': 1,
            'residual': True
        }
    }

# Output base dir
base_dir = "configs_split_experiments/3DInfomax/_test"
os.makedirs(base_dir, exist_ok=True)

# Generate config files
for dataset in datasets:
    dataset_dir = os.path.join(base_dir, dataset)
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
            
