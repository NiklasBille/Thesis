import sys
import subprocess
import argparse

from train import get_arguments

p = argparse.ArgumentParser()
p.add_argument('--config', type=str)
p.add_argument('--device', type=str, default='cuda:0')
args = p.parse_args()

if not args.config:
    p.error('No config found, add --config')
else:
    config_path = args.config

run_args = get_arguments()

processes = []
for seed in run_args.multiple_seeds:
    cmd = [
        "python",
        "train.py", 
        "--config",
        config_path,
        "--seed",
        str(seed),
        "--device",
        args.device
    ]
    print(f"Starting process for seed {seed}: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd)
    processes.append(proc)

# Wait for all parallel processes to finish
for proc in processes:
    proc.wait()

print("All runs completed.")


