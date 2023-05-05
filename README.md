# Competitive deep learning for enhancing image reconstruction and classification of individuals with stable and progressive mild cognitive impairment

This work is currently ongoing.

# Introduction

This is the repo for the model proposed in the paper. Briefly, the model performs medical image (scans) reconstruction. The result shows that the reconstructed scans not only have better image quality, but also improves the prediction accuracy of MRI progression (sMRI vs pMRI).

<p align="center">
<img src="figures/architecture.png" width="695"/>
</p>

The model is trained on a dataset from ADNI and evaluated on an exteranl dataset from NACC.

MRI progression prediction performance:

| Input Images | CNR | SNR | SSIM | BRISQUE | PIQE |
|--------------|-----|-----|------|---------|------|
| **(a). Results on ADNI test partition** | | | | | |
| Original | 2.111±0.274 | 0.686±0.052 | - | 43.513±1.594 | 41.936±2.062 |
| Diced | 0.726±0.131 | 0.537±0.046 | 0.319±0.065 | 43.567±0.333 | 78.307±5.322 |
| GAN-VAN | 2.235±0.299 | 1.857±0.214 | 0.544±0.051 | 42.485±0.741 | 72.028±3.068 |
| GAN-NOV | 1.934±0.335 | 1.437±0.395 | 0.580±0.059 | 42.246±0.563 | 68.120±1.351 |
| **(b). Results on NACC cohort** | | | | | |
| Original | 2.065±0.294 | 0.676±0.074 | - | 44.292±2.940 | 43.182±6.298 |
| Diced | 0.696±0.156 | 0.526±0.074 | 0.348±0.108 | 43.742±1.256 | 78.442±4.695 |
| GAN-VAN | 2.193±0.359 | 1.821±0.230 | 0.523±0.105 | 42.563±1.128 | 72.133±4.024 |
| GAN-NOV | 1.790±0.363 | 1.432±0.534 | 0.553±0.116 | 42.505±1.066 | 68.176±2.079 |

Image quality metrics:

<p align="center">
<img src="figures/iqs.png" width="850"/>
</p>

See the paper for additional information

## Quick start

1. CUBLAS_WORKSPACE_CONFIG=:4096:8 python rcgan_main.py   (Train and generate scans for single G)
2. CUBLAS_WORKSPACE_CONFIG=:4096:8 python rcgans_main.py  (Train and generate scans for multiple G)
3. CUBLAS_WORKSPACE_CONFIG=:4096:8 python classifier_main.py    (Evaluate prediction performance using CNN)
4. (optional, in plot/) python plot.py
5. (optional, image quality) python image_quality.py
6. (optional, MCC) python matrix_stat.py

## Environments

1. Install python3
2. Install the environments.yml (Anaconda environment)
3. (optional, image quality) Install matlab for python (if standard method not work, try: sudo python setup.py install --prefix="/home/xzhou/anaconda3/envs/py36/")

## Data Preprocessing

The data preprocessing follows a similar procedure as in this work: ()

## Hyper-parameter Tuning

The json files that contains 'config' name (i.e. config.json) are the files that can be used to modify most of the hyperparameters.
