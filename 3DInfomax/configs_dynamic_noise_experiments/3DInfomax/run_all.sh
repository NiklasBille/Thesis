#!/bin/bash

# bace
python run_multiple.py --config=configs_noise_experiments/3DInfomax/bace/noise=0.0.yml
python run_multiple.py --config=configs_noise_experiments/3DInfomax/bace/noise=0.05.yml
python run_multiple.py --config=configs_noise_experiments/3DInfomax/bace/noise=0.1.yml
python run_multiple.py --config=configs_noise_experiments/3DInfomax/bace/noise=0.2.yml


# bbbp
python run_multiple.py --config=configs_noise_experiments/3DInfomax/bbbp/noise=0.0.yml
python run_multiple.py --config=configs_noise_experiments/3DInfomax/bbbp/noise=0.05.yml
python run_multiple.py --config=configs_noise_experiments/3DInfomax/bbbp/noise=0.1.yml
python run_multiple.py --config=configs_noise_experiments/3DInfomax/bbbp/noise=0.2.yml

# clintox
python run_multiple.py --config=configs_noise_experiments/3DInfomax/clintox/noise=0.0.yml
python run_multiple.py --config=configs_noise_experiments/3DInfomax/clintox/noise=0.05.yml
python run_multiple.py --config=configs_noise_experiments/3DInfomax/clintox/noise=0.1.yml
python run_multiple.py --config=configs_noise_experiments/3DInfomax/clintox/noise=0.2.yml

# esol
python run_multiple.py --config=configs_noise_experiments/3DInfomax/esol/noise=0.0.yml
python run_multiple.py --config=configs_noise_experiments/3DInfomax/esol/noise=0.05.yml
python run_multiple.py --config=configs_noise_experiments/3DInfomax/esol/noise=0.1.yml
python run_multiple.py --config=configs_noise_experiments/3DInfomax/esol/noise=0.2.yml

# freesolv
python run_multiple.py --config=configs_noise_experiments/3DInfomax/freesolv/noise=0.0.yml
python run_multiple.py --config=configs_noise_experiments/3DInfomax/freesolv/noise=0.05.yml
python run_multiple.py --config=configs_noise_experiments/3DInfomax/freesolv/noise=0.1.yml
python run_multiple.py --config=configs_noise_experiments/3DInfomax/freesolv/noise=0.2.yml

# hiv
python run_multiple.py --config=configs_noise_experiments/3DInfomax/hiv/noise=0.0.yml
python run_multiple.py --config=configs_noise_experiments/3DInfomax/hiv/noise=0.05.yml
python run_multiple.py --config=configs_noise_experiments/3DInfomax/hiv/noise=0.1.yml
python run_multiple.py --config=configs_noise_experiments/3DInfomax/hiv/noise=0.2.yml

# lipo
python run_multiple.py --config=configs_noise_experiments/3DInfomax/lipo/noise=0.0.yml
python run_multiple.py --config=configs_noise_experiments/3DInfomax/lipo/noise=0.05.yml
python run_multiple.py --config=configs_noise_experiments/3DInfomax/lipo/noise=0.1.yml
python run_multiple.py --config=configs_noise_experiments/3DInfomax/lipo/noise=0.2.yml

# sider
python run_multiple.py --config=configs_noise_experiments/3DInfomax/sider/noise=0.0.yml
python run_multiple.py --config=configs_noise_experiments/3DInfomax/sider/noise=0.05.yml
python run_multiple.py --config=configs_noise_experiments/3DInfomax/sider/noise=0.1.yml
python run_multiple.py --config=configs_noise_experiments/3DInfomax/sider/noise=0.2.yml

# tox21
python run_multiple.py --config=configs_noise_experiments/3DInfomax/tox21/noise=0.0.yml
python run_multiple.py --config=configs_noise_experiments/3DInfomax/tox21/noise=0.05.yml
python run_multiple.py --config=configs_noise_experiments/3DInfomax/tox21/noise=0.1.yml
python run_multiple.py --config=configs_noise_experiments/3DInfomax/tox21/noise=0.2.yml

# toxcast
python run_multiple.py --config=configs_noise_experiments/3DInfomax/toxcast/noise=0.0.yml
python run_multiple.py --config=configs_noise_experiments/3DInfomax/toxcast/noise=0.05.yml
python run_multiple.py --config=configs_noise_experiments/3DInfomax/toxcast/noise=0.1.yml
python run_multiple.py --config=configs_noise_experiments/3DInfomax/toxcast/noise=0.2.yml
