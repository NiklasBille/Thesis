# Reproducing Uni-Mol
This folder reproduces the results from https://github.com/deepmodeling/Uni-Mol

### Running this repo
Ensure you have the "dptechnology/unimol:latest-pytorch1.11.0-cuda11.3" docker image available. To get this image run:

    docker pull dptechnology/unimol:latest-pytorch1.11.0-cuda11.3
To create a docker image from the dockerfile, navigate to /Thesis/Uni-Mol and run:
    
    docker build --tag unimol unimol/docker
Run the new image as a container while loading this directory into the container. We also ensure that the setup.py file has been run:

    docker run --gpus all --rm -it --runtime=nvidia -v $(pwd):/workspace -w /workspace unimol bash -c "cd unimol && python setup.py install && bash"