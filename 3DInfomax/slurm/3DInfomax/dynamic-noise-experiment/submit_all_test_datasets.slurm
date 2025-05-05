#!/bin/bash -l
#SBATCH --account p200709
#SBATCH --time 00:10:00
#SBATCH --partition gpu
#SBATCH --qos dev
#SBATCH --nodes 1
#SBATCH --job-name=ds_test_%A_%a

# Load Apptainer
module load Apptainer/1.3.4-GCCcore-13.3.0

# Get dataset name from array index
DATASET=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" slurm/3DInfomax/flip-pertubation/datasets.txt)
echo ">>> Starting run for dataset: $DATASET"

# Pull the image if not already present
if [ ! -f 3dinfomax-complete.sif ]; then
    apptainer pull 3dinfomax-complete.sif docker://niklasbille/3dinfomax-complete-image:latest
fi

# Run everything inside the container
apptainer exec --nv --bind $PWD:/workspace 3dinfomax-complete.sif bash -c "
    cd /workspace && \
    pip install ogb==1.3.6 && \
    python noise_experiment/main.py && \
    pip install ogb==1.3.3 && \
    gpu_counter=0 && \
    for noise in 0.0 0.05 0.1 0.2; do
        config=configs_noise_experiments/3DInfomax/_test/$DATASET/noise=\$noise.yml
        gpu=\$((gpu_counter % 4))
        echo Running \$config on cuda:\$gpu
        python run_multiple.py --config=\$config --device=cuda:\$gpu &
        ((gpu_counter++))
    done && \
    wait
"
