#!/bin/bash -l
#SBATCH --account p200709
#SBATCH --time 05:00:00
#SBATCH --partition gpu
#SBATCH --qos default
#SBATCH --nodes 1
#SBATCH --gpus-per-node 4

# Load Apptainer
module load Apptainer/1.3.4-GCCcore-13.3.0

# Get dataset name from array index
DATASET=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" slurm/GraphCL/medium_datasets.txt)
echo ">>> Starting run for dataset: $DATASET"

# Pull the image if not already present
if [ ! -f 3dinfomax-complete.sif ]; then
    apptainer pull 3dinfomax-complete.sif docker://niklasbille/3dinfomax-complete-image:latest
fi

# Run everything inside the container
apptainer exec --nv --bind $PWD:/workspace 3dinfomax-complete.sif bash -c "
    cd /workspace && \
    gpu_counter=0 && \
    # Loop over the different training proportions
    for prop in 0.6 0.7 0.8; do
        config=configs_split_experiments/GraphCL/$DATASET/scaff/train_prop=\$prop.yml
        gpu=\$((gpu_counter % 3))  # use 0-3 GPUs
        echo Running SCAFF \$config on cuda:\$gpu
        python -u run_multiple.py --config=\$config --device=cuda:\$gpu &
        ((gpu_counter++))
    done
    wait  # Wait for all scaff runs to finish
"
