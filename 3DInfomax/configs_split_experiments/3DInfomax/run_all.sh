#!/bin/bash

# bace
python run_multiple.py --config=configs_split_experiments/3DInfomax/bace/scaff/train_prop=0.6.yml
python run_multiple.py --config=configs_split_experiments/3DInfomax/bace/scaff/train_prop=0.7.yml
python run_multiple.py --config=configs_split_experiments/3DInfomax/bace/scaff/train_prop=0.8.yml

python run_multiple.py --config=configs_split_experiments/3DInfomax/bace/random/train_prop=0.6.yml
python run_multiple.py --config=configs_split_experiments/3DInfomax/bace/random/train_prop=0.7.yml
python run_multiple.py --config=configs_split_experiments/3DInfomax/bace/random/train_prop=0.8.yml

# bbbp 
# NOTE: we can not use train_prop=0.6 on bbbp since the produced validation set contains no positives
#       which makes auc undefined due to division with zero. Thus we drop this specific setting.
# python run_multiple.py --config=configs_split_experiments/3DInfomax/bbbp/scaff/train_prop=0.6.yml 
# python run_multiple.py --config=configs_split_experiments/3DInfomax/bbbp/scaff/train_prop=0.7.yml
# python run_multiple.py --config=configs_split_experiments/3DInfomax/bbbp/scaff/train_prop=0.8.yml

# python run_multiple.py --config=configs_split_experiments/3DInfomax/bbbp/random/train_prop=0.6.yml
# python run_multiple.py --config=configs_split_experiments/3DInfomax/bbbp/random/train_prop=0.7.yml
# python run_multiple.py --config=configs_split_experiments/3DInfomax/bbbp/random/train_prop=0.8.yml

# clintox
python run_multiple.py --config=configs_split_experiments/3DInfomax/clintox/scaff/train_prop=0.6.yml
python run_multiple.py --config=configs_split_experiments/3DInfomax/clintox/scaff/train_prop=0.7.yml
python run_multiple.py --config=configs_split_experiments/3DInfomax/clintox/scaff/train_prop=0.8.yml

python run_multiple.py --config=configs_split_experiments/3DInfomax/clintox/random/train_prop=0.6.yml
python run_multiple.py --config=configs_split_experiments/3DInfomax/clintox/random/train_prop=0.7.yml
python run_multiple.py --config=configs_split_experiments/3DInfomax/clintox/random/train_prop=0.8.yml

# esol
python run_multiple.py --config=configs_split_experiments/3DInfomax/esol/scaff/train_prop=0.6.yml
python run_multiple.py --config=configs_split_experiments/3DInfomax/esol/scaff/train_prop=0.7.yml
python run_multiple.py --config=configs_split_experiments/3DInfomax/esol/scaff/train_prop=0.8.yml

python run_multiple.py --config=configs_split_experiments/3DInfomax/esol/random/train_prop=0.6.yml
python run_multiple.py --config=configs_split_experiments/3DInfomax/esol/random/train_prop=0.7.yml
python run_multiple.py --config=configs_split_experiments/3DInfomax/esol/random/train_prop=0.8.yml

# freesolv
python run_multiple.py --config=configs_split_experiments/3DInfomax/freesolv/scaff/train_prop=0.6.yml
python run_multiple.py --config=configs_split_experiments/3DInfomax/freesolv/scaff/train_prop=0.7.yml
python run_multiple.py --config=configs_split_experiments/3DInfomax/freesolv/scaff/train_prop=0.8.yml

python run_multiple.py --config=configs_split_experiments/3DInfomax/freesolv/random/train_prop=0.6.yml
python run_multiple.py --config=configs_split_experiments/3DInfomax/freesolv/random/train_prop=0.7.yml
python run_multiple.py --config=configs_split_experiments/3DInfomax/freesolv/random/train_prop=0.8.yml

# hiv
# python run_multiple.py --config=configs_split_experiments/3DInfomax/hiv/scaff/train_prop=0.6.yml
# python run_multiple.py --config=configs_split_experiments/3DInfomax/hiv/scaff/train_prop=0.7.yml
# python run_multiple.py --config=configs_split_experiments/3DInfomax/hiv/scaff/train_prop=0.8.yml

# python run_multiple.py --config=configs_split_experiments/3DInfomax/hiv/random/train_prop=0.6.yml
# python run_multiple.py --config=configs_split_experiments/3DInfomax/hiv/random/train_prop=0.7.yml
# python run_multiple.py --config=configs_split_experiments/3DInfomax/hiv/random/train_prop=0.8.yml

# lipo
python run_multiple.py --config=configs_split_experiments/3DInfomax/lipo/scaff/train_prop=0.6.yml
python run_multiple.py --config=configs_split_experiments/3DInfomax/lipo/scaff/train_prop=0.7.yml
python run_multiple.py --config=configs_split_experiments/3DInfomax/lipo/scaff/train_prop=0.8.yml

python run_multiple.py --config=configs_split_experiments/3DInfomax/lipo/random/train_prop=0.6.yml
python run_multiple.py --config=configs_split_experiments/3DInfomax/lipo/random/train_prop=0.7.yml
python run_multiple.py --config=configs_split_experiments/3DInfomax/lipo/random/train_prop=0.8.yml

# sider
python run_multiple.py --config=configs_split_experiments/3DInfomax/sider/scaff/train_prop=0.6.yml
python run_multiple.py --config=configs_split_experiments/3DInfomax/sider/scaff/train_prop=0.7.yml
python run_multiple.py --config=configs_split_experiments/3DInfomax/sider/scaff/train_prop=0.8.yml

python run_multiple.py --config=configs_split_experiments/3DInfomax/sider/random/train_prop=0.6.yml
python run_multiple.py --config=configs_split_experiments/3DInfomax/sider/random/train_prop=0.7.yml
python run_multiple.py --config=configs_split_experiments/3DInfomax/sider/random/train_prop=0.8.yml

# tox21
python run_multiple.py --config=configs_split_experiments/3DInfomax/tox21/scaff/train_prop=0.6.yml
python run_multiple.py --config=configs_split_experiments/3DInfomax/tox21/scaff/train_prop=0.7.yml
python run_multiple.py --config=configs_split_experiments/3DInfomax/tox21/scaff/train_prop=0.8.yml

python run_multiple.py --config=configs_split_experiments/3DInfomax/tox21/random/train_prop=0.6.yml
python run_multiple.py --config=configs_split_experiments/3DInfomax/tox21/random/train_prop=0.7.yml
python run_multiple.py --config=configs_split_experiments/3DInfomax/tox21/random/train_prop=0.8.yml

# toxcast
python run_multiple.py --config=configs_split_experiments/3DInfomax/toxcast/scaff/train_prop=0.6.yml
python run_multiple.py --config=configs_split_experiments/3DInfomax/toxcast/scaff/train_prop=0.7.yml
python run_multiple.py --config=configs_split_experiments/3DInfomax/toxcast/scaff/train_prop=0.8.yml

python run_multiple.py --config=configs_split_experiments/3DInfomax/toxcast/random/train_prop=0.6.yml
python run_multiple.py --config=configs_split_experiments/3DInfomax/toxcast/random/train_prop=0.7.yml
python run_multiple.py --config=configs_split_experiments/3DInfomax/toxcast/random/train_prop=0.8.yml
