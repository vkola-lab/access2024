# rcgan

Generative Adversarial Network for 3D Object Reconstruction & Classificatioon

This is the repo for 3d_RCGAN

## Quick start

1. CUBLAS_WORKSPACE_CONFIG=:4096:8 python rcgan_main.py   (Train and generate scans for single G)
2. CUBLAS_WORKSPACE_CONFIG=:4096:8 python rcgans_main.py  (Train and generate scans for multiple G)
3. CUBLAS_WORKSPACE_CONFIG=:4096:8 python classifier_main.py    (Evaluate performance using CNN)

## Environments

1. Install python3
2. Install the environments.yml

## Configuration

The config.json is the file that can be used to easily modify most of the hyperparameters.

