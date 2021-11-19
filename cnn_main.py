# main file for CNN classifier
# Created: 9/2/2021
# Status: in progress
# CUBLAS_WORKSPACE_CONFIG=:4096:8 python cnn_main.py

import sys
import torch

import numpy as np

from networks import CNN_Wrapper
from utils import read_json
from tabulate import tabulate


def CNN(model_name, config, Wrapper, num_exps):
    print('Evaluation metric: {}'.format(config['loss_metric']))
    reports = []
    for exp_idx in range(num_exps):
        net = Wrapper(config, model_name, exp_idx*1000)
        # cnn.load('./checkpoint_dir/{}_exp{}/'.format('cnn_mri_pre', 0), fixed=False)
        net.train(epochs = config['train_epochs'], training_prints=2)
        report = net.test(out=True,key='test')
        reports.append(report)

    res = [model_name]
    accs = [r['accuracy'] for r in reports]
    res += ['%.3f+-%.3f' % (np.mean(accs), np.std(accs))]
    pms = [r['macro avg']['precision'] for r in reports]
    pws = [r['weighted avg']['precision'] for r in reports]
    res += ['%.3f+-%.3f' % (np.mean(pms), np.std(pms))]
    res += ['%.3f+-%.3f' % (np.mean(pws), np.std(pws))]
    rms = [r['macro avg']['recall'] for r in reports]
    rws = [r['weighted avg']['recall'] for r in reports]
    res += ['%.3f+-%.3f' % (np.mean(rms), np.std(rms))]
    res += ['%.3f+-%.3f' % (np.mean(rws), np.std(rws))]
    fms = [r['macro avg']['f1-score'] for r in reports]
    fws = [r['weighted avg']['f1-score'] for r in reports]
    res += ['%.3f+-%.3f' % (np.mean(fms), np.std(fms))]
    res += ['%.3f+-%.3f' % (np.mean(fws), np.std(fws))]

    return res


def main():
    num_exps = 3
    table = []
    table.append(['Model', 'Accuracy', 'Precision (macro avg)', 'Precision (weighted avg)', 'Recall (macro avg)', 'Recall (weighted avg)', 'F1-score (macro avg)', 'F1-score (weighted avg)'])
    torch.use_deterministic_algorithms(True)
    print('-'*100)
    print('Running CNN classifier for AD status; Dataset: T (Original ADNI)')
    config = read_json('./config.json')['cnn_T']
    model_name = 'CNN_{}_T'.format(config['loss_metric'])
    res = CNN(model_name, config, CNN_Wrapper, num_exps)
    table.append(res)
    print('-'*100)
    print('-'*100)
    print('Running CNN classifier for AD status; Dataset: Z (Sliced ADNI)')
    config = read_json('./config.json')['cnn_Z']
    model_name = 'CNN_{}_Z'.format(config['loss_metric'])
    res = CNN(model_name, config, CNN_Wrapper, num_exps)
    table.append(res)
    print('-'*100)
    print('-'*100)
    print('Running CNN classifier for AD status; Dataset: G (Generated ADNI)')
    config = read_json('./config.json')['cnn_G']
    model_name = 'CNN_{}_G'.format(config['loss_metric'])
    res = CNN(model_name, config, CNN_Wrapper, num_exps)
    table.append(res)
    print('-'*100)
    print(tabulate(table, headers='firstrow'))
    print('-'*100)
    print('Completed')


if __name__ == "__main__":
    main()
