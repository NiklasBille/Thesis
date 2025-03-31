# Reproducing Uni-Mol
This folder reproduces the results from https://github.com/deepmodeling/Uni-Mol

## Running this repo
Ensure you have the "dptechnology/unimol:latest-pytorch1.11.0-cuda11.3" docker image available. To get this image run:

    docker pull dptechnology/unimol:latest-pytorch1.11.0-cuda11.3
To create a docker image from the dockerfile, navigate to /Thesis/Uni-Mol and run:
    
    docker build --tag unimol unimol/docker
Run the new image as a container while loading this directory into the container. We also ensure that the setup.py file has been run:

    docker run --gpus all --rm -it --runtime=nvidia -v $(pwd):/workspace -w /workspace unimol bash -c "cd unimol && python setup.py install && bash"

### Downloading data
To download the data and unpack it run:

    mkdir -p data
    wget --directory-prefix=data https://bioos-hermite-beijing.tos-cn-beijing.volces.com/unimol_data/finetune/molecular_property_prediction.tar.gz 
    tar -xvzf data/molecular_property_prediction.tar.gz -C data/
    rm data/molecular_property_prediction.tar.gz

To use random splits, see `data/README.md`.

### Downloading weights
To download pre-trained weights for both all hydrogen and no hydrogen case run:

    mkdir -p weights/pretrained 
    wget --directory-prefix=weights/pretrained https://github.com/deepmodeling/Uni-Mol/releases/download/v0.1/mol_pre_all_h_220816.pt
    wget --directory-prefix=weights/pretrained https://github.com/deepmodeling/Uni-Mol/releases/download/v0.1/mol_pre_no_h_220816.pt

### Results folders
Create folders for results:

    mkdir -p weights/finetuned
    mkdir -p results

### Training 
All hyperparameters can be found in the Uni-Mol repo. To train on e.g. FreeSolv using a single GPU run:

    data_path="/workspace/unimol/data/molecular_property_prediction"  # replace to your data path
    save_dir="/workspace/unimol/weights/finetuned/freesolv"  # replace to your save path
    dict_name="dict.txt"
    weight_path="/workspace/unimol/weights/pretrained/mol_pre_all_h_220816.pt"  # replace to your ckpt path
    task_name="freesolv"  # molecular property prediction task name 
    task_num=1
    loss_func=finetune_mse
    lr=8e-5
    batch_size=64
    epoch=60
    dropout=0.2
    warmup=0.1
    local_batch_size=64
    only_polar=-1
    conf_size=11
    seed=0

    if [ "$task_name" == "qm7dft" ] || [ "$task_name" == "qm8dft" ] || [ "$task_name" == "qm9dft" ]; then
    	metric="valid_agg_mae"
    elif [ "$task_name" == "esol" ] || [ "$task_name" == "freesolv" ] || [ "$task_name" == "lipo" ]; then
        metric="valid_agg_rmse"
    else 
        metric="valid_agg_auc"
    fi

    export NCCL_ASYNC_ERROR_HANDLING=1
    export OMP_NUM_THREADS=1
    export CUDA_VISIBLE_DEVICES=1 # which device to use
    update_freq=`expr $batch_size / $local_batch_size`
    python $(which unicore-train) $data_path --task-name $task_name --user-dir /workspace/unimol --train-subset train --valid-subset valid,test \
           --conf-size $conf_size \
           --num-workers 8 --ddp-backend=c10d \
           --dict-name $dict_name \
           --task mol_finetune --loss $loss_func --arch unimol_base  \
           --classification-head-name $task_name --num-classes $task_num \
           --optimizer adam --adam-betas "(0.9, 0.99)" --adam-eps 1e-6 --clip-norm 1.0 \
           --lr-scheduler polynomial_decay --lr $lr --warmup-ratio $warmup --max-epoch $epoch --batch-size $local_batch_size --pooler-dropout $dropout\
           --update-freq $update_freq --seed $seed \
           --fp16 --fp16-init-scale 4 --fp16-scale-window 256 \
           --log-interval 100 --log-format simple \
           --validate-interval 1 --tensorboard-logdir $save_dir/tsb \
           --finetune-from-model $weight_path \
           --best-checkpoint-metric $metric --patience 20 \
           --save-dir $save_dir --only-polar $only_polar


To evaluate run:

     data_path="/workspace/unimol/data/molecular_property_prediction"  # replace to your data path
     results_path=/workspace/unimol/results  # replace to your results path
     weight_path="/workspace/unimol/weights/finetuned/freesolv/checkpoint_best.pt"  # replace to your ckpt path
     batch_size=64
     task_name='freesolv' # data folder name 
     task_num=1
     loss_func='finetune_mse'
     dict_name='dict.txt'
     conf_size=11
     only_polar=-1

    export CUDA_VISIBLE_DEVICES=1 # which device to use

    python /workspace/unimol/unimol/infer.py --user-dir /workspace/unimol $data_path --task-name $task_name --valid-subset test \
           --results-path $results_path \
           --num-workers 8 --ddp-backend=c10d --batch-size $batch_size \
           --conf-size $conf_size \
           --task mol_finetune --loss $loss_func --arch unimol_base \
           --classification-head-name $task_name --num-classes $task_num \
           --dict-name $dict_name \
           --only-polar $only_polar  \
           --path $weight_path  \
           --fp16 --fp16-init-scale 4 --fp16-scale-window 256 \
           --log-interval 50 --log-format simple 

### Metrics
To obtain metrics and write them to a csv file use the script `getEvaluationMetricsCSV.py`. Currently this is only implemented for regression tasks, but we can extend to classification tasks when needed.

### Tensorboard
By including the argument `--tensorboard-logdir $save_dir/tsb` we can inspect loss curves etc. Outside of the container run the command `tensorboard --logdir=<path_to_tsb>` and follow the link.

