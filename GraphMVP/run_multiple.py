import os
import sys
import argparse

import subprocess

# TODO: pass device correctly.

p = argparse.ArgumentParser()
p.add_argument('--config', type=str)
p.add_argument('--device', type=str, default='cuda:0', help="Device to use: 'cpu', '0' for cuda:0, '1' for cuda:1, etc.")
args = p.parse_args()

if not args.config:
    p.error('No config found, add --config')
else:
    config_path = os.path.join('/workspace/', args.config)

multiple_seeds = [0, 1]

#dataset_task_type = {'freesolv': 'regression', 'bace': 'classification'}
task_type = 'regression'

train_script = 'molecule_finetune_regression.py' if task_type == 'regression' else 'molecule_finetune.py'
src_folder = 'src_regression' if task_type == 'regression' else 'src_classification.py'
processes = []

for seed in multiple_seeds:
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

