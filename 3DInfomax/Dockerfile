# Base image with CUDA support
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/opt/conda/bin:$PATH"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    git \
    curl \
    ca-certificates \
    libglib2.0-0 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda.sh && \
    bash /miniconda.sh -b -p /opt/conda && \
    rm /miniconda.sh

# Create conda environment
COPY environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml && rm /tmp/environment.yml

# Set the default shell to bash and activate conda environment
SHELL ["/bin/bash", "-c"]
RUN echo "conda activate 3DInfomax" >> ~/.bashrc
ENV CONDA_DEFAULT_ENV=3DInfomax
ENV PATH="/opt/conda/envs/3DInfomax/bin:$PATH"

# Set working directory
WORKDIR /workspace

# Install additional dependencies inside the container build
RUN conda install -y \
    pytorch-cuda=11.8 -c pytorch -c nvidia && \
    pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.4.0+cu118.html && \
    pip install torch_geometric && \
    pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/cu118/repo.html && \
    pip install icecream tensorboard pyaml seaborn

# Default command
CMD ["/bin/bash"]
