#!/bin/bash -l
#SBATCH --account=p200709
#SBATCH --time=00:05:00
#SBATCH --partition=gpu
#SBATCH --qos=dev
#SBATCH --nodes=1
#SBATCH --job-name=bace_random_0.6

module load Apptainer/1.3.4-GCCcore-13.3.0

# Pull image if needed
if [ ! -f 3dinfomax-complete.sif ]; then
    apptainer pull 3dinfomax-complete.sif docker://niklasbille/3dinfomax-complete-image:latest
fi


# Run inside container
apptainer exec --nv --bind $PWD:/workspace 3dinfomax-complete.sif bash -c "
    cd /workspace && \
    echo Running config for bace 0 noise && \
    python run_multiple.py --config=configs_noise_experiments/3DInfomax/bace/noise=0.0.yml --device=cuda:0
"
