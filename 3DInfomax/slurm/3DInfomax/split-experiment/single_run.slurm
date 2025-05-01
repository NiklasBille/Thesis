#!/bin/bash -l
#SBATCH --account p200709
#SBATCH --time 01:30:00
#SBATCH --partition gpu
#SBATCH --qos default
#SBATCH --nodes 1
#SBATCH --gpus-per-node 4

# Load Apptainer
module load Apptainer/1.3.4-GCCcore-13.3.0

# Pull the image if not already present
if [ ! -f 3dinfomax-complete.sif ]; then
    apptainer pull 3dinfomax-complete.sif docker://niklasbille/3dinfomax-complete-image:latest
fi

# Run inside the container
apptainer exec --nv --bind $PWD:/workspace 3dinfomax-complete.sif bash -c "
    cd /workspace && \
    echo Running config for freesolv random 0.8 && \
    python -u run_multiple.py --config=configs_split_experiments/3DInfomax/freesolv/random/train_prop=0.8.yml --device=cuda:0
"
