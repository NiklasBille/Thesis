# Reproducing 3DInfomax
This folder reproduces the results from https://github.com/HannesStark/3DInfomax

### Running this repo
Ensure you have the "nvidia/cuda:11.8.0-devel-ubuntu22.04" docker image available. To get this image run:

    docker pull nvidia/cuda:11.8.0-devel-ubuntu22.04
Create a docker image from the dockerfile:
    
    docker build -t 3dinfomax-complete-image .
Run the new image as a container, while loading this directory into the container:

    docker run --gpus all --rm -it --runtime=nvidia -v $(pwd):/workspace -w /workspace 3dinfomax-complete-image

If you wanna fine-tune on QM9_homo run:

    python train.py --config=configs_clean/tune_QM9_homo.yml
If you wanna fine-tune on freesolve run:

    python train.py --config=configs_clean/tune_freesolv.yml

Sometimes errors regrading torch shapes can occur. Reinstalling ogb==1.3.3 has fixed this issue for me.

ogb is funky: if you get a weird error related to this library try doing the following:

    pip uninstall ogb
    pip install ogb
    pip install ogb==1.3.3

## Changes made to original repo
Since the original paper is from 2022, a few changes was needed to find a set of dependency versions that all can collaborate.

* New versions of torch_geometrics removed the swish function, therefore in spherical_message_passing l. 22-23, a new swish function has been defined, with the same functionality.
* l. 8 in train.py has been changed to "from ogb.lsc import PCQM4MDataset, PCQM4MEvaluator" since ogb.lsc renamed the dataset. 
* l. 4 in utils.py has been changed to "from collections.abc import MutableMapping". MutableMapping is now imported from collections.abc instead of collections as specified in original repository.
* The requirements.txt file has been changed to specifiy new versions. This file should always be used in collaboration with the Dockerfile. It can not be used to create a conda env, since not all dependency versions are specified here. 
* A dockerfile has been implemented to create an image that should make it possible for anyone to run the code.
* In `dataset/geom_drugs_dataset.py` the function `torch.linalg.eigh` had an unexpected argument `eigenvectors=True`. This was removed since the function defaults to also return the eigenvectors.
* In `train.py` there is a mistake in the the function `train_ogbg`. When using random splitting (`if args.force_random_split == True:`) they store indices as train/train/train when it should be train/val/test.
* Customized the Tensorboard logger to include all relevant metrics and overlay these on the dashboard. 
    * In `train.py` we now also pass `test_loader` on when we call `trainer.train()`.
    * In `trainer.py` we created a custom layout for the Tensorboard which includes specified metrics and main metric. We also set a flag (`val=True`) when computing metrics on train set since this allows us to also log the main metric on the train set.


## Noise experiment 
Currently working on adding noise to the fine-tuning dataset of QM9. The folder noise_experiment contains a flip_pertubation_noise.py file that implements this file. Currently these methods are used in qm9_dataset.py when initializing the data_dict. For testing purposes we currently just add the noise to the atoms and edges for the first molecule.
Also created a pre-train_QM9_small_test.yml script file for testing the noise experiment on a small version of QM9 first. 

When extracting features for the different ogb/moleculenet datasets, it can require you to install ogb==1.3.6 first.