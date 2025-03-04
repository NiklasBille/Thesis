FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

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

ENV PATH="/opt/conda/bin:$PATH"

# Create conda environment
COPY environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml && rm /tmp/environment.yml

# Set default shell to bash and activate the environment
SHELL ["/bin/bash", "-c"]
RUN echo "source activate 3DInfomax" > ~/.bashrc
ENV CONDA_DEFAULT_ENV=3DInfomax
ENV PATH="/opt/conda/envs/3DInfomax/bin:$PATH"

# Set working directory
WORKDIR /workspace

CMD ["/bin/bash"]
