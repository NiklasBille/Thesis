import os
import shutil

base_root = "3DInfomax/runs/static_noise/3DInfomax"
target_root = "Results/noise/3DInfomax"

# Metrics to extract
allowed_metrics = {"mae", "rmse", "prcauc", "rocauc", "BCEWithLogitsLoss", "OGBNanLabelBCEWithLogitsLoss", "L1Loss", "OGBNanLabelloss"}
loss_alias = ["BCEWithLogitsLoss", "OGBNanLabelBCEWithLogitsLoss", "L1Loss", "OGBNanLabelloss"]
# Walk through all datasets
for dataset in os.listdir(base_root):
    dataset_path = os.path.join(base_root, dataset)
    if not os.path.isdir(dataset_path):
        continue

    # Walk through all noise levels
    for noise_level in os.listdir(dataset_path):
        noise_path = os.path.join(dataset_path, noise_level)
        if not os.path.isdir(noise_path):
            continue

        # Destination directory
        target_noise_path = os.path.join(target_root, dataset, noise_level)
        os.makedirs(target_noise_path, exist_ok=True)

        # Walk through all seeds (subfolders with actual runs)
        sorted_folders = sorted([
            d for d in os.listdir(noise_path)
            if os.path.isdir(os.path.join(noise_path, d))
        ])

        for i, folder_name in enumerate(sorted_folders, start=1):
            folder_path = os.path.join(noise_path, folder_name)
            seed_folder = os.path.join(target_noise_path, f"seed{i}")
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
                    with open(os.path.join(seed_folder, "evaluation.txt"), "w") as out_f:
                        for line in filtered_lines:
                            if any(alias in line for alias in loss_alias):
                                # Replace the alias with "loss"
                                for alias in loss_alias:
                                    line = line.replace(alias, "loss")
                            out_f.write(f'test_{line}')
                        # out_f.write("\n")

            # Filter and copy evaluation_train.txt
            train_path = os.path.join(folder_path, "evaluation_train.txt")
            if os.path.exists(train_path):
                filtered_lines = []
                with open(train_path, "r") as f:
                    for line in f:
                        key = line.split(":")[0].strip()
                        if key in allowed_metrics:
                            filtered_lines.append(line)

                if filtered_lines:
                    # Append
                    with open(os.path.join(seed_folder, "evaluation.txt"), "a") as out_f:
                        for line in filtered_lines:
                            if any(alias in line for alias in loss_alias):
                                # Replace the alias with "loss"
                                for alias in loss_alias:
                                    line = line.replace(alias, "loss")
                            out_f.write(f'train_{line}')
                        # out_f.write("\n")

            # Filter and copy evaluation_val.txt
            val_path = os.path.join(folder_path, "evaluation_val_best_checkpoint.txt")
            if os.path.exists(val_path):
                filtered_lines = []
                with open(val_path, "r") as f:
                    for line in f:
                        key = line.split(":")[0].strip()
                        if key in allowed_metrics:
                            filtered_lines.append(line)

                if filtered_lines:
                    # Append
                    with open(os.path.join(seed_folder, "evaluation.txt"), "a") as out_f:
                        for line in filtered_lines:
                            if any(alias in line for alias in loss_alias):
                                # Replace the alias with "loss"
                                for alias in loss_alias:
                                    line = line.replace(alias, "loss")
                            out_f.write(f'val_{line}')

            # Copy events file if present
            for fname in os.listdir(folder_path):
                if fname.startswith("events.out.tfevents"):
                    shutil.copy(
                        os.path.join(folder_path, fname),
                        os.path.join(seed_folder, fname)
                    )