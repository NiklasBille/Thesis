#!/bin/bash -l
#SBATCH --account p200709
#SBATCH --time 36:00:00
#SBATCH --partition gpu
#SBATCH --qos default
#SBATCH --nodes 1
#SBATCH --job-name=ds_test_%A_%a

# Load Apptainer
module load Apptainer/1.3.4-GCCcore-13.3.0

# Get dataset name from array index
echo ">>> Starting run for dataset: muv"

# Pull the image if not already present
if [ ! -f 3dinfomax-complete.sif ]; then
    apptainer pull 3dinfomax-complete.sif docker://niklasbille/3dinfomax-complete-image:latest
fi

# Run everything inside the container
apptainer exec --nv --bind $PWD:/workspace 3dinfomax-complete.sif bash -c "
    cd /workspace && \
    gpu_counter=0 && \
    for noise in 0.0 0.05 0.1 0.2; do
        config=configs_static_noise_experiments/3DInfomax/muv/noise=\$noise.yml
        gpu=\$((gpu_counter % 4))
        echo Running \$config on cuda:\$gpu
        python run_multiple.py --config=\$config --device=cuda:\$gpu &
        ((gpu_counter++))
    done && \
    wait
"
