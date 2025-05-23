# Use NVIDIA CUDA base image
FROM nvidia/cuda:11.0.3-devel-ubuntu20.04

# update apt and get miniconda
RUN apt-get update \
    && apt-get install -y \
    wget \
    git \
    && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh


# install miniconda
ENV PATH="/root/miniconda3/bin:$PATH"
RUN mkdir /root/.conda && bash Miniconda3-latest-Linux-x86_64.sh -b

# Set the working directory
#WORKDIR /workspace

# Clone the GraphMVP repository
#RUN git clone https://github.com/chao1224/GraphMVP.git

# create conda environment and install packages
RUN conda init bash \
    && . ~/.bashrc \
    && conda create -n GraphMVP python=3.7 -y \
    && conda activate GraphMVP \
    && conda install -y -c rdkit rdkit \
    && conda install conda-forge::ogb=1.3.5 \
    && pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html \
    && conda install -y numpy networkx scikit-learn \ 
    && pip install ase \
    && pip install git+https://github.com/bp-kelley/descriptastorus \
    && wget https://data.pyg.org/whl/torch-1.10.0%2Bcu113/torch_cluster-1.5.9-cp37-cp37m-linux_x86_64.whl \
    && pip install torch_cluster-1.5.9-cp37-cp37m-linux_x86_64.whl \
    && wget https://data.pyg.org/whl/torch-1.10.0%2Bcu113/torch_scatter-2.0.9-cp37-cp37m-linux_x86_64.whl \
    && pip install torch_scatter-2.0.9-cp37-cp37m-linux_x86_64.whl \
    && wget https://data.pyg.org/whl/torch-1.10.0%2Bcu113/torch_sparse-0.6.12-cp37-cp37m-linux_x86_64.whl \
    && pip install torch_sparse-0.6.12-cp37-cp37m-linux_x86_64.whl \
    && pip install torch-geometric==1.7.2 \
    && pip install "pyaml<21.10.0" \
    && pip install tensorboard

RUN echo "conda activate GraphMVP" >> ~/.bashrc