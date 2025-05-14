# SLURM 
To run the experiments using SLURM do the following:
    cd /project/home/p200709/Niklas_Rasmus_thesis/Thesis/3DInfomax

To test everythin works before training do:

    sbatch --array=0-1 slurm/3DInfomax/flip-pertubation/submit_all_test_datasets.slurm

If everything works start training all experiments:

    sbatch --array=0-9 slurm/3DInfomax/flip-pertubation/submit_all_datasets.slurm

You can monitor the progress using 
    squeue -u $USER


## Running interactively on single GPU cluster
    
    # Go to project directory
    cd /project/home/p200709/Niklas_Rasmus_thesis/Thesis/3DInfomax

    # Create a gpu node for 1 hour 
    salloc -A p200709 -t 00:20:00 -p gpu -q dev -N 1
    
    # Get apptainer module
    module load Apptainer/1.3.4-GCCcore-13.3.0

    # Get image
    apptainer pull 3dinfomax-complete.sif docker://niklasbille 3dinfomax-complete-image:latest

    # Open container binding current workspace
    apptainer shell --nv --bind $PWD:/workspace 3dinfomax-complete.sif

    # cd workspace 
    cd /workspace/

    # fix ogb
    pip install ogb==1.3.6

    # Load datasets
    python noise_experiment/main.py

    # Downgrade ogb again 
    pip install ogb==1.3.3


    # Run experiment
    chmod +x slurm/3DInfomax/flip-pertubation/test_run_all.sh
    ./slurm/3DInfomax/flip-pertubation/test_run_all.sh

If you dont wanna run test change the parameters