# Reproducing GraphMVP
This folder reproduces the results from https://github.com/chao1224/GraphMVP

### Running this repo
Ensure you have the "nvidia/cuda:11.0.3-devel-ubuntu22.04" docker image available. To get this image run:

    docker pull nvidia/cuda:11.0.3-devel-ubuntu20.04
Create a docker image from the dockerfile, navigate to /Thesis/GraphMVP and run::
    
    docker build --tag graphmvp docker/
Run the new image as a container, while loading this directory into the container:

        docker run --gpus all --rm -it --runtime=nvidia -v $(pwd):/workspace -w /workspace graphmvp

## Downloading data from paper
To download the dataset used in the paper run:

    wget --directory-prefix=datasets/ http://snap.stanford.edu/gnn-pretrain/data/chem_dataset.zip
    python -m zipfile -e datasets/chem_dataset.zip datasets/
    mv datasets/dataset datasets/molecule_datasets
    rm datasets/chem_dataset.zip

To obtain the correct processed data features we need to re-process the datasets, since they are not processed in the desired way after downloading them. Therefore, first run the `noise_experiment/process_datasets.py`

    python noise_experiment/process_datasets.py

## Training
To run the training script for e.g. lipo, cd into src_regression and run:

    save_dir="/workspace/results/lipo_seed2"
    weight_path="/workspace/weights/pretrained/GraphMVP_regression.pth"
    dataset=lipophilicity
    device=1
    seed=2

    python molecule_finetune_regression.py --device $device \
    	--seed $seed --runseed $seed \
    	--input_model_file $weight_path \
        --output_model_dir $save_dir \
        --dataset $dataset

For all the downstream tasks the author of GraphMVP did not tune the hyperparameters. 


## Changes made
* cu111 instead of cu102 or cu110 due to our hardware (not compatible with version cu102 and cu110 has been removed from the link in original repo)
* We must install ogb=1.3.5 before torch otherwise the environment uses a  PyTorch installation with CUDA10.2 for some reason. See [this link](https://discuss.pytorch.org/t/geforce-rtx-3090-with-cuda-capability-sm-86-is-not-compatible-with-the-current-pytorch-installation/123499) for information on this issue.
* When loading data outside of the train script (e.g. when inspecting datasets) we would run into issues when loading torch_geometric: 
    ```
    ImportError: /lib/x86_64-linux-gnu/libstdc++.so.6: version GLIBCXX_3.4.29' not found
    ```
     
    The issue could be that the installed SciPy build (1.7.3) requires a newer C++ standard library (a newer libstdc++), but the system’s libstdc++ is too old to include the GLIBCXX_3.4.29 version. One possible solution is running `conda install scipy=1.7.0 -c conda-forge` which hopefully does not break things since 1.7.1-1.7.3 only comes with bug-fixes.
* In `src_regression/molecule_finetune_regression.py` we implemented a Tensorboard logger. We modified the `eval` function to include an argument `compute_loss=False` that computes and returns the loss (no grad) when set to `True`. This is done so that we can log the validation and test loss.
