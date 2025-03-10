# Use NVIDIA CUDA base image
FROM nvidia/cuda:11.0.3-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Update package lists and install system dependencies
RUN apt-get update && apt-get install -y \
    python3.8 python3.8-dev python3.8-venv \
    python3-pip \
    git wget curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Ensure Python 3.8 is the default version
RUN ln -sf /usr/bin/python3.8 /usr/bin/python && ln -sf /usr/bin/python3.8 /usr/bin/python3

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install required Python packages
RUN pip install --no-cache-dir \
    numpy \
    pandas \
    networkx \
    "paddlepaddle>=2.0.0" \
    "pgl>=2.1.5" \
    rdkit-pypi \
    scikit-learn

# Set the working directory
WORKDIR /workspace

# Default command
CMD ["/bin/bash"]
