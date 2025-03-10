# Use the existing UniMol image as the base
FROM dptechnology/unimol:latest-pytorch1.11.0-cuda11.3

# Set the working directory
WORKDIR /workspace

# Copy the setup.py and other necessary files (if needed)
COPY setup.py /workspace/

# Install dependencies from setup.py
RUN pip install --no-cache-dir .

# Run the setup script
RUN python setup.py install
RUN pip install rdkit-pypi==2022.9.3

# Install ipykernel for jupyter notebook support
RUN pip3 install ipykernel

