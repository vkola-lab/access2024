# main file for 3D Reconstruction GAN
# Created: 6/16/2021
# Status: ok
# CUBLAS_WORKSPACE_CONFIG=:4096:8 python rgan_main.py

import sys
import torch
import wandb

import numpy as np

from networks import RGAN_Wrapper
from utils import read_json


SWEEP = 1

def RGAN(model_name, config, Wrapper):
    print('Loss metric: {}'.format(config['loss_metric']))
    net = Wrapper(config, model_name, SWEEP)
    net.train(epochs = config['train_epochs'])
    if not SWEEP:
        net.load(fixed=False)
        net.generate(datas=[net.all_dataloader], whole=True, samples=False)


def main():
    torch.use_deterministic_algorithms(True)

    if SWEEP:
        config_default = read_json('./config.json')['rgan']
        wandb.init(config=config_default)
        config = wandb.config
    else:
        print('-'*100)
        print('Running 3D Reconstruction GAN (3D-RGAN)')
        config = read_json('./config.json')['rgan']
            
    model_name = str(SWEEP)+'_RGAN_{}'.format(config['loss_metric'])
    RGAN(model_name, config, RGAN_Wrapper)

    if not SWEEP:
        print('-'*100)
        print('Completed')


if __name__ == "__main__":
    if SWEEP:
        print('performing parameters searching...')
        sweep_config = read_json('./config.json')['sweep_rgan']
        sweep_id = wandb.sweep(sweep_config, project='gan')
        wandb.agent(sweep_id, main)
    else:
        main()
