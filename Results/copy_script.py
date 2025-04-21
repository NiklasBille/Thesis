import os
import shutil
import re

# Base directory containing the experiment result folders
base_dir = "3DInfomax/runs/flip-pertubation/3DInfomax/esol/noise=0.05"
target_base_dir = "test_folder"

os.makedirs(target_base_dir, exist_ok=True)

# Get all subdirectories sorted alphabetically, which should correspond to the seed values
sorted_folders = sorted([
    d for d in os.listdir(base_dir)
    if os.path.isdir(os.path.join(base_dir, d))
])

for i, folder_name in enumerate(sorted_folders, start=1):
    folder_path = os.path.join(base_dir, folder_name)
    seed_folder = os.path.join(target_base_dir, f"seed{i}")
    os.makedirs(seed_folder, exist_ok=True)

    # Copy evaluation_test.txt
    eval_path = os.path.join(folder_path, "evaluation_test.txt")
    if os.path.exists(eval_path):
        shutil.copy(eval_path, os.path.join(seed_folder, "evaluation_test.txt"))

    # Copy events file
    for fname in os.listdir(folder_path):
        if fname.startswith("events.out.tfevents"):
            shutil.copy(
                os.path.join(folder_path, fname),
                os.path.join(seed_folder, fname)
            )
