# Reproducing Uni-Mol
This folder reproduces the results from https://github.com/deepmodeling/Uni-Mol/tree/main/unimol

### Running this repo

### Downloading data from paper
To download the dataset used in the paper run:

    wget --directory-prefix=datasets/ http://snap.stanford.edu/gnn-pretrain/data/chem_dataset.zip
    python -m zipfile -e datasets/chem_dataset.zip datasets/
    mv datasets/dataset datasets/molecule_datasets
    rm datasets/chem_dataset.zip

### Changes in environment
