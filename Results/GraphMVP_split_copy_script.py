import os
import shutil

base_root = "GraphMVP/runs/split/GraphMVP"
target_root = "Results/split/GraphMVP"

# Metrics to extract
allowed_metrics = {"mae", "rmse", "prcauc", "rocauc"}

# Walk through all datasets
for dataset in os.listdir(base_root):
    dataset_path = os.path.join(base_root, dataset)
    if not os.path.isdir(dataset_path):
        continue

    # Walk through all split types (e.g., random, scaffold)
    for split_type in os.listdir(dataset_path):
        split_type_path = os.path.join(dataset_path, split_type)
        if not os.path.isdir(split_type_path):
            continue

        # Walk through all split parameters (e.g., train_prop=0.6)
        for split_param in os.listdir(split_type_path):
            split_param_path = os.path.join(split_type_path, split_param)
            if not os.path.isdir(split_param_path):
                continue

            # Destination path
            target_split_path = os.path.join(target_root, dataset, split_type, split_param)
            os.makedirs(target_split_path, exist_ok=True)

            # Walk through all run folders (seeds)
            sorted_folders = sorted([
                d for d in os.listdir(split_param_path)
                if os.path.isdir(os.path.join(split_param_path, d))
            ])

            for i, folder_name in enumerate(sorted_folders, start=1):
                folder_path = os.path.join(split_param_path, folder_name)
                seed_folder = os.path.join(target_split_path, f"seed{i}")
                os.makedirs(seed_folder, exist_ok=True)

                # Filter and copy evaluation_test.txt
                eval_path = os.path.join(folder_path, "evaluation_test.txt")
                if os.path.exists(eval_path):
                    filtered_lines = []
                    with open(eval_path, "r") as f:
                        for line in f:
                            key = line.split(":")[0].strip()
                            if key in allowed_metrics:
                                filtered_lines.append(line)

                    if filtered_lines:
                        with open(os.path.join(seed_folder, "evaluation_test.txt"), "w") as out_f:
                            out_f.writelines(filtered_lines)

                # Copy events file if present
                for fname in os.listdir(folder_path):
                    if fname.startswith("events.out.tfevents"):
                        shutil.copy(
                            os.path.join(folder_path, fname),
                            os.path.join(seed_folder, fname)
                        )
