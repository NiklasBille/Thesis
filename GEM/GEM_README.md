# GEM
Since GEM is integrated into PaddleHelix and uses function outside of its own directory, we have to copy the whole framework.

    #SSH 
    git clone git@github.com:PaddlePaddle/PaddleHelix.git

    #HTTPS
    git clone https://github.com/PaddlePaddle/PaddleHelix.git



In case the original authors update the GEM directory, these experiments should work with commit d7d9b39.

## Environment
To create a docker image and then run it as a container run the following. 
First ensure the nvidia/cuda:11.0.3-devel-ubuntu20.04 base image is available

    docker pull nvidia/cuda:11.0.3-devel-ubuntu20.04

Creating the dockerfile imediately will return an error due to the PaddleHelix library has changed without updating GEM. Therefore change l. 37 in src/utils to:

    import paddle
    fluid = paddle.static

Create a docker image from the dockerfile

    docker build -t gem .

Create a container and move the PaddleHelix repo into workspace

    docker run --gpus all --rm -it --runtime=nvidia -v $(pwd):/workspace -w /workspace gem


Then navigate into /apps/pretrained_compunds/ChemRL/GEM/


## Pretraining
Use the following command to download the demo data which is a tiny subset [Zinc Dataset](https://zinc.docking.org/) and run pretrain tasks.

    bash scripts/pretrain.sh

Note that the data preprocessing step will be time-consuming since it requires running MMFF optimization for all molecules. The demo data will take several hours to finish in a single V100 GPU card. The pretrained model will be save under `./pretrain_models`.

We also provide our pretrained model [here](https://baidu-nlp.bj.bcebos.com/PaddleHelix/pretrained_models/compound/pretrain_models-chemrl_gem.tgz) for reproducing the downstream finetuning results. Also, the pretrained model can be used for other molecular property prediction tasks.

## Downstream finetuning
After the pretraining, the downstream tasks can use the pretrained model as initialization. 

Firstly, download the pretrained model from the previous step:

    wget https://baidu-nlp.bj.bcebos.com/PaddleHelix/pretrained_models/compound/pretrain_models-chemrl_gem.tgz
    tar xzf pretrain_models-chemrl_gem.tgz

Download the downstream molecular property prediction datasets from [MoleculeNet](http://moleculenet.ai/), including classification tasks and regression tasks:

    wget https://baidu-nlp.bj.bcebos.com/PaddleHelix/datasets/compound_datasets/chemrl_downstream_datasets.tgz
    tar xzf chemrl_downstream_datasets.tgz
    
Run downstream finetuning and the final results will be saved under `./log/pretrain-$dataset/final_result`. 

    # classification tasks
    bash scripts/finetune_class.sh
    # regression tasks
    bash scripts/finetune_regr.sh

The whole finetuning process for all datasets requires 1-2 days in a single V100 GPU card.