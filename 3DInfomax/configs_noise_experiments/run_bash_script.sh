#!/bin/bash

# Esol
python run_multiple.py --config=configs_noise_experiments/GraphCL/esol/GraphCL_esol_noise=0.1.yml
python run_multiple.py --config=configs_noise_experiments/GraphCL/esol/GraphCL_esol_noise=0.2.yml
python run_multiple.py --config=configs_noise_experiments/GraphCL/esol/GraphCL_esol_noise=0.05.yml
python run_multiple.py --config=configs_noise_experiments/GraphCL/esol/GraphCL_esol_noise=0.yml

# Freesolv
python run_multiple.py --config=configs_noise_experiments/GraphCL/freesolv/GraphCL_freesolv_noise=0.1.yml
python run_multiple.py --config=configs_noise_experiments/GraphCL/freesolv/GraphCL_freesolv_noise=0.2.yml
python run_multiple.py --config=configs_noise_experiments/GraphCL/freesolv/GraphCL_freesolv_noise=0.05.yml
python run_multiple.py --config=configs_noise_experiments/GraphCL/freesolv/GraphCL_freesolv_noise=0.yml

# Lipo
python run_multiple.py --config=configs_noise_experiments/GraphCL/lipo/GraphCL_lipo_noise=0.1.yml
python run_multiple.py --config=configs_noise_experiments/GraphCL/lipo/GraphCL_lipo_noise=0.2.yml
python run_multiple.py --config=configs_noise_experiments/GraphCL/lipo/GraphCL_lipo_noise=0.05.yml
python run_multiple.py --config=configs_noise_experiments/GraphCL/lipo/GraphCL_lipo_noise=0.yml