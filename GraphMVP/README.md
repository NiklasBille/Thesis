# Reproducing GraphMVP
This folder reproduces the results from https://github.com/chao1224/GraphMVP

### Running this repo
Ensure you have the "nvidia/cuda:11.0.3-devel-ubuntu22.04" docker image available. To get this image run:

    docker pull nvidia/cuda:11.0.3-devel-ubuntu20.04
Create a docker image from the dockerfile:
    
    docker build --tag graphmvp:pytorch1.9.1-cuda11.0.3 GraphMVP/
Run the new image as a container, while loading this directory into the container:

    docker run --gpus all --rm -it --runtime=nvidia -v <path_to_graphmvp_repo>:/workspace -w /workspace graphmvp:pytorch1.9.0-cuda11.0.3

### Downloading data from paper
To download the dataset used in the paper run:

    wget --directory-prefix=datasets/ http://snap.stanford.edu/gnn-pretrain/data/chem_dataset.zip
    python -m zipfile -e datasets/chem_dataset.zip datasets/
    mv datasets/dataset datasets/molecule_datasets
    rm datasets/chem_dataset.zip



### Changes in environment
- cu111 instead of cu102 or cu110 due to our hardware (not compatible with version cu102 and cu110 has been removed from the link)
- need to install ogb=1.3.5 before torch otherwise the environment uses a  PyTorch installation with CUDA10.2 for some reason.
https://discuss.pytorch.org/t/geforce-rtx-3090-with-cuda-capability-sm-86-is-not-compatible-with-the-current-pytorch-installation/123499