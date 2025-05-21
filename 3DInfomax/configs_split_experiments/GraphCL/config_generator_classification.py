import os
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

# Define datasets and associated task counts
dataset_dict = {
    'hiv': {'tasks': 1, 'loss_func': 'BCEWithLogitsLoss', 'batch_size': 30},
    'bace': {'tasks': 1, 'loss_func': 'BCEWithLogitsLoss', 'batch_size': 30},
    'bbbp': {'tasks': 1, 'loss_func': 'BCEWithLogitsLoss', 'batch_size': 30},
    'tox21': {'tasks': 12, 'loss_func': 'OGBNanLabelBCEWithLogitsLoss', 'batch_size': 30},
    'toxcast': {'tasks': 617, 'loss_func': 'OGBNanLabelBCEWithLogitsLoss', 'batch_size': 30},
    'sider': {'tasks': 27, 'loss_func': 'OGBNanLabelBCEWithLogitsLoss', 'batch_size': 32},
    'clintox': {'tasks': 2, 'loss_func': 'OGBNanLabelBCEWithLogitsLoss', 'batch_size': 30},
    'muv': {'tasks': 17, 'loss_func': 'OGBNanLabelBCEWithLogitsLoss', 'batch_size': 30},
}

datasets = list(dataset_dict.keys())
split_types = ['scaff', 'random']
train_props = [0.8, 0.7, 0.6]

# Base config structure
def create_config(dataset, train_prop, split_type):
    tasks = dataset_dict[dataset]['tasks']
    loss_func = dataset_dict[dataset]['loss_func']
    batch_size = dataset_dict[dataset]['batch_size']
    force_random_split = True if split_type == 'random' else False
    return {
        'experiment_name': f'GraphCL_{dataset}_{split_type}={train_prop}',
        'logdir': f'runs/split/GraphCL/{dataset}/{split_type}/train_prop={train_prop}',
        'multiple_seeds': [1, 2, 3], 
        'train_prop': train_prop, 
        'force_random_split': force_random_split,
        'pretrain_checkpoint': 'runs/PNA_graphcl_drugs_smaller/best_checkpoint.pt',
        'transfer_layers': ['gnn.'],
        'dataset': f'ogbg-mol{dataset}',
        'num_epochs': 1000,
        'batch_size': batch_size,
        'log_iterations': 30,
        'patience': 60,
        'minimum_epochs': 120,
        'loss_func': loss_func,
        'required_data': ['dgl_graph', 'targets'],
        'metrics': ['prcauc', 'rocauc'],
        'optimizer': 'Adam',
        'optimizer_params': {
            'lr': 0.001
        },
        'scheduler_step_per_batch': False,
        'lr_scheduler': 'WarmUpWrapper',
        'lr_scheduler_params': {
            'warmup_steps': FlowList([700, 700, 350]),
            'interpolation': 'linear',
            'wrapped_scheduler': 'ReduceLROnPlateau',
            'factor': 0.5,
            'patience': 25,
            'min_lr': 0.000001,
            'mode': 'min',
            'verbose': True
        },
        'model_type': 'PNA',
        'model_parameters': {
            'target_dim': tasks,
            'hidden_dim': 50,
            'mid_batch_norm': True,
            'last_batch_norm': True,
            'readout_batchnorm': True,
            'batch_norm_momentum': 0.1,
            'readout_hidden_dim': 50,
            'readout_layers': 2,
            'dropout': 0.0,
            'propagation_depth': 3,
            'aggregators': ['mean', 'max', 'min', 'std'],
            'scalers': ['identity', 'amplification', 'attenuation'],
            'readout_aggregators': ['min', 'max', 'mean', 'sum'],
            'pretrans_layers': 2,
            'posttrans_layers': 1,
            'residual': True
        }
    }

# Output base dir
base_dir = "configs_split_experiments/GraphCL"
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
