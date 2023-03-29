# main file for 3D Reconstruction GAN
# Created: 10/7/2021
# Status: ok
# CUBLAS_WORKSPACE_CONFIG=:4096:8 python rcgan_main.py

import sys
import torch
import wandb

import numpy as np

from networks import RCGAN_Wrapper
from networks import CNN_Wrapper
from utils import read_json
from tabulate import tabulate
from make_figures import figures

SWEEP = 0

def CNN(model_name, config, Wrapper, num_exps, train=True):
    print('Evaluation metric: {}'.format(config['loss_metric']))
    reports = [[],[],[],[]]
    for exp_idx in range(num_exps):
        net = Wrapper(config, model_name, exp_idx)
        # net.visualize()
        # print('loading model...')
        net.load('./checkpoint_dir/CNN_Standard_Pre0/', fixed=False)
        if train:
            net.train(epochs = config['train_epochs'], training_prints=2)
        else:
            net.load(net.checkpoint_dir)
        # net.visualize(prefix=model_name)
        reports[0].append(net.test_b(out=True,key='test'))
        reports[1].append(net.test_b(out=True,key='ext'))
        reports[2].append(net.test_b(out=True,key='train'))
        reports[3].append(net.test_b(out=True,key='valid'))
    ress = [[model_name], [model_name+'_E'], [model_name+'_Train'], [model_name+'_Valid']]

    for rep, res in zip(reports, ress):
        accs = [r['accuracy'] for r in rep]
        res += ['%.3f+-%.3f' % (np.mean(accs), np.std(accs))]
        pws = [r['weighted avg']['precision'] for r in rep]
        res += ['%.3f+-%.3f' % (np.mean(pws), np.std(pws))]
        rws = [r['weighted avg']['recall'] for r in rep]
        res += ['%.3f+-%.3f' % (np.mean(rws), np.std(rws))]
        fws = [r['weighted avg']['f1-score'] for r in rep]
        res += ['%.3f+-%.3f' % (np.mean(fws), np.std(fws))]
        if SWEEP:
            wandb.log({"f1score":np.mean(fws)})
        # fms = [r['macro avg']['f1-score'] for r in rep]
        # res += ['%.3f+-%.3f' % (np.mean(fms), np.std(fms))]

    return ress

def CNN_main():
    num_exps = 5
    table = []
    table.append(['Model', 'Accuracy', 'Precision (weighted avg)', 'Recall (weighted avg)', 'F1-score (weighted avg)'])
    torch.use_deterministic_algorithms(True)
    data_root = '/data1/RGAN_Data/'
    datasets = ['CG_1']
    
    # train = False
    train = True
    for d in datasets:
        print('-'*100)
        config = read_json('./config.json')['cnn_E']
        config['data_dir'] = data_root+d+'/'
        # model_name = 'CNN_'+d
        model_name = 'CNN_{}_'.format(config['loss_metric'])+d
        ress = CNN(model_name, config, CNN_Wrapper, num_exps, train)
        table.append(ress[0])
        table.append(ress[1])
        print('-'*100)
    print(tabulate(table, headers='firstrow'))

def RCGAN(model_name, config, Wrapper):
    print('Loss metric: {}'.format(config['loss_metric']))
    net = Wrapper(config, model_name, SWEEP)
    if 1:
        figures(['torchviz', 'hiddenlayer', 'netron', 'tensorboard'], net, net.train_dataloader)
    # net.train(epochs = config['train_epochs'])
    # net.generate(datas=[net.train_dataloader, net.valid_dataloader, net.test_dataloader, net.ext_dataloader], whole=True, samples=True, ext=True) #all & ext are same slices
    # net.generate(datas=[net.ext_dataloader], whole=True, samples=True, ext=True) #all & ext are same slices
    # net.generate(datas=[net.all_dataloader, net.ext_dataloader], whole=True, samples=True, ext=True) #all & ext are same slices
    # print('generated')
    # CNN_main()


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
    print('-'*100)

    if not SWEEP:
        print('Completed')


if __name__ == "__main__":
    if SWEEP:
        print('[performing parameters searching...]')
        sweep_config = read_json('./config.json')['sweep_rgan']
        sweep_id = wandb.sweep(sweep_config, project='RCGAN-22')
        wandb.agent(sweep_id, main, count=100)
    else:
        main()
