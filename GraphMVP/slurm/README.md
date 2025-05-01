# Slurm

To run experiments using batch jobs, first ensure you stand in the GraphMVP folder.
To test everything works before training run the test scripts:

    sbatch --array=0-9 slurm/split/test_random_all.slurm
    sbatch --array=0-9 slurm/split/test_scaff_all.slurm
    sbatch --array=0-9 slurm/flip-pertubation/test_all.slurm

If everything works run the full experiments with different arguments


## Running interactively 
On GraphMVP we need to do the following:

    # Create a gpu node for 20 minutes 
    salloc -A p200709 -t 00:20:00 -p gpu -q dev -N 1
    
    # Get apptainer module
    module load Apptainer/1.3.4-GCCcore-13.3.0

    # Pull image from dockerhub
    apptainer pull graphmvp.sif docker://niklasbille/graphmvp:latest

    # Open container with image
    apptainer shell --nv --bind $PWD:/workspace graphmvp.sif

    # cd workspace
    cd /workspace/

    # Activate GraphMVP conda env
    source /root/miniconda3/etc/profile.d/conda.sh
    conda activate GraphMVP