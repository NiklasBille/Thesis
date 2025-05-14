#!/bin/bash -l
#SBATCH --account p200709
#SBATCH --time 07:00:00
#SBATCH --partition gpu
#SBATCH --qos default
#SBATCH --nodes 1
#SBATCH --job-name=GraphCL_seed_4to7_noise

# Load Apptainer
module load Apptainer/1.3.4-GCCcore-13.3.0

# Get dataset name from array index
DATASET=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" slurm/GraphCL/regression_datasets.txt)
echo ">>> Starting run for dataset: $DATASET"

# Pull the image if not already present
if [ ! -f 3dinfomax-complete.sif ]; then
    apptainer pull 3dinfomax-complete.sif docker://niklasbille/3dinfomax-complete-image:latest
fi

# Run everything inside the container
apptainer exec --nv --bind $PWD:/workspace 3dinfomax-complete.sif bash -c "
    cd /workspace && \
    gpu_counter=0 && \
    for noise in 0.0 0.05 0.1 0.2; do
        config=configs_static_noise_experiments/GraphCL/$DATASET/noise=\$noise.yml
        gpu=\$((gpu_counter % 4))
        echo Running \$config on cuda:\$gpu
        python run_multiple.py --config=\$config --device=cuda:\$gpu &
        ((gpu_counter++))
    done && \
    wait
"
