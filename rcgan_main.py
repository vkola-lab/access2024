# main file for 3D Reconstruction GAN
# Created: 10/7/2021
# Status: ok
# CUBLAS_WORKSPACE_CONFIG=:4096:8 python rcgan_main.py

import sys
import torch
import wandb

import numpy as np

from networks import RCGAN_Wrapper
from utils import read_json


SWEEP = 0

def RCGAN(model_name, config, Wrapper):
    print('Loss metric: {}'.format(config['loss_metric']))
    net = Wrapper(config, model_name, SWEEP)
    net.train(epochs = config['train_epochs'])

    if not SWEEP:
        net.load(fixed=False)
        net.generate(datas=[net.all_dataloader, net.ext_dataloader], whole=True, samples=False, ext=True)
        print('outputs generated')


def main():
    torch.use_deterministic_algorithms(True)

    if SWEEP:
        config_default = read_json('./config.json')['rgan']
        wandb.init(config=config_default)
        config = wandb.config
    else:
        print('-'*100)
        print('Running 3D Reconstruction & Classification GAN (3D-RCGAN)')
        config = read_json('./config.json')['rgan']

    model_name = str(SWEEP)+'_RCGAN_{}'.format(config['loss_metric'])
    RCGAN(model_name, config, RCGAN_Wrapper)

    if not SWEEP:
        print('-'*100)
        print('Completed')


if __name__ == "__main__":
    if SWEEP:
        print('[performing parameters searching...]')
        sweep_config = read_json('./config.json')['sweep_rgan']
        sweep_id = wandb.sweep(sweep_config, project='rcgan_sp')
        wandb.agent(sweep_id, main, count=100)
    else:
        main()
