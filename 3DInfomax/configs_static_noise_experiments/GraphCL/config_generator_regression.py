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

# Define datasets and noise levels
datasets = ['esol', 'freesolv', 'lipo']
noise_levels = [0.0, 0.05, 0.1, 0.2]

# Base config structure
def create_config(dataset, noise):
    return {
        'experiment_name': f'GraphCL_{dataset}_static_noise={noise}',
        'logdir': f'runs/static_noise/GraphCL/{dataset}/noise={noise}', 
        'multiple_seeds': [4, 5, 6],  
        'pretrain_checkpoint': 'runs/PNA_graphcl_drugs_smaller/best_checkpoint.pt',
        'transfer_layers': ['gnn.'],
        'dataset': f'ogbg-mol{dataset}',
        'noise_level': noise,
        'dynamic_noise': False,
        'num_epochs': 1000, 
        'batch_size': 30,
        'log_iterations': 30,
        'patience': 60,
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
base_dir = "configs_static_noise_experiments/GraphCL"
os.makedirs(base_dir, exist_ok=True)

# Generate config files
for dataset in datasets:
    dataset_dir = os.path.join(base_dir, dataset)
    os.makedirs(dataset_dir, exist_ok=True)
    for noise in noise_levels:
        config = create_config(dataset, noise)
        filename = os.path.join(dataset_dir, f"noise={noise}.yml")
        with open(filename, 'w') as f:
            yaml.dump(config, f, Dumper=CustomDumper, sort_keys=False, default_flow_style=False)
        print(f"Created: {filename}")
