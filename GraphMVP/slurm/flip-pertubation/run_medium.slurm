#!/bin/bash -l
#SBATCH --account p200709
#SBATCH --time 07:00:00
#SBATCH --partition gpu
#SBATCH --qos default
#SBATCH --nodes 1
#SBATCH --gpus-per-node 4

# Load Apptainer
module load Apptainer/1.3.4-GCCcore-13.3.0

# Get dataset name from array index
DATASET=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" slurm/medium_datasets.txt)
echo ">>> Starting run for dataset: $DATASET"

# Pull the image if not already present
if [ ! -f 3dinfomax-complete.sif ]; then
    apptainer pull graphmvp.sif docker://niklasbille/graphmvp:latest
fi

# Run everything inside the container
apptainer exec --nv --bind $PWD:/workspace graphmvp.sif bash -c "
    cd /workspace && \
    source /root/miniconda3/etc/profile.d/conda.sh
    conda activate GraphMVP
    gpu_counter=0 && \
    # Loop over the different training proportions
    for noise in 0.0 0.05 0.1 0.2; do
        config=configs_split_experiments/GraphMVP/_test/$DATASET/noise=\$noise.yml
        gpu=\$((gpu_counter % 3))  # use 0-3 GPUs
        echo Running RANDOM \$config on cuda:\$gpu
        python -u run_multiple.py --config=\$config --device=cuda:\$gpu &
        ((gpu_counter++))
    done
    wait  # Wait for all random runs to finish
"
