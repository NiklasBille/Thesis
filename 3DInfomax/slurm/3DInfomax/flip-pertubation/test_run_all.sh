#!/bin/bash

datasets=("bace" "bbbp" "clintox" "esol" "freesolv" "hiv" "lipo" "sider" "tox21" "toxcast")
noises=(0.0 0.05 0.1 0.2)

gpus=(0 1 2 3)
i=0

for dataset in "${datasets[@]}"; do
    for noise in "${noises[@]}"; do
        gpu=${gpus[$((i % 4))]}
        config="configs_noise_experiments/3DInfomax/_test/${dataset}/noise=${noise}.yml"

        echo "Launching $config on GPU $gpu"
        python run_multiple.py --config="$config" --device="cuda:$gpu" &

        ((i++))

        # Run in batches of 4
        if (( i % 4 == 0 )); then
            wait  # Wait for all 4 to finish
        fi
    done
done

wait  # Final sync
