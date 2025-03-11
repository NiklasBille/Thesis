# Reproducing Uni-Mol
This folder reproduces the results from https://github.com/deepmodeling/Uni-Mol

### Running this repo
Ensure you have the "dptechnology/unimol:latest-pytorch1.11.0-cuda11.3" docker image available. To get this image run:

    docker pull dptechnology/unimol:latest-pytorch1.11.0-cuda11.3
To create a docker image from the dockerfile, navigate to /Thesis/Uni-Mol and run:
    
    docker build --tag unimol unimol/docker
Run the new image as a container while loading this directory into the container. We also ensure that the setup.py file has been run:

    docker run --gpus all --rm -it --runtime=nvidia -v $(pwd):/workspace -w /workspace unimol bash -c "cd unimol && python setup.py install && bash"

### Downloading data
To download the data and unpack it run:

    mkdir -p data
    wget --directory-prefix=data https://bioos-hermite-beijing.tos-cn-beijing.volces.com/unimol_data/finetune/molecular_property_prediction.tar.gz 
    tar -xvzf data/molecular_property_prediction.tar.gz -C data/
    rm data/molecular_property_prediction.tar.gz

### Downloading weights
To download pre-trained weights for both all hydrogen and no hydrogen case run:

    mkdir -p weights/pretrained 
    wget --directory-prefix=weights/pretrained https://github.com/deepmodeling/Uni-Mol/releases/download/v0.1/mol_pre_all_h_220816.pt
    wget --directory-prefix=weights/pretrained https://github.com/deepmodeling/Uni-Mol/releases/download/v0.1/mol_pre_no_h_220816.pt

### Results folders
Create folders for results:

    mkdir -p weights/finetuned
    mkdir -p results


