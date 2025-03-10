# Reproducing Uni-Mol
This folder reproduces the results from https://github.com/deepmodeling/Uni-Mol/tree/main/unimol

### Running this repo
Ensure you have the "dptechnology/unimol:latest-pytorch1.11.0-cuda11.3" docker image available. To get this image run:

    docker pull dptechnology/unimol:latest-pytorch1.11.0-cuda11.3
Create a docker image from the dockerfile:
    
    docker build --tag del:del Uni-Mol/unimol
Run the new image as a container, while loading this directory into the container:

    docker run --gpus all --rm -it --runtime=nvidia -v <path_to_graphmvp_repo>:/workspace -w /workspace graphmvp:pytorch1.9.0-cuda11.0.3

### Downloading data from paper
To download the dataset used in the paper run:

    wget --directory-prefix=datasets/ http://snap.stanford.edu/gnn-pretrain/data/chem_dataset.zip
    python -m zipfile -e datasets/chem_dataset.zip datasets/
    mv datasets/dataset datasets/molecule_datasets
    rm datasets/chem_dataset.zip

### Changes in environment
