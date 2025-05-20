import os
import sys
import argparse
import subprocess

sys.path.insert(0, 'src_classification')
from config import args as runargs

p = argparse.ArgumentParser()
p.add_argument('--config', type=str)
p.add_argument('--device', type=str, default='cuda:0', help="Device to use: 'cpu', '0' for cuda:0, '1' for cuda:1, etc.")
args = p.parse_args()

if not args.config:
    p.error('No config found, add --config')
else:
    config_path = os.path.join('/workspace/', args.config)

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
    'muv': 'classification'
}

assert runargs.dataset in dataset_task_type.keys(), f"{runargs.dataset} not valid dataset."

task_type = dataset_task_type[runargs.dataset]
train_script = 'molecule_finetune_regression.py' if task_type == 'regression' else 'molecule_finetune.py'
src_folder = 'src_regression' if task_type == 'regression' else 'src_classification'

processes = []
for seed in runargs.multiple_seeds:
    cmd = [
        "python",
        train_script, 
        "--config",
        config_path,
        "--runseed",
        str(seed),
        "--device",
        args.device
    ]

    print(f"Starting process for seed {seed}: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, cwd=os.path.join('/workspace/', src_folder))
    processes.append(proc)

# Wait for all parallel processes to finish
for proc in processes:
    proc.wait()

print("All runs completed.")

