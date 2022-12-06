# main file for CNN classifier
# Updated: 12/25/2021
# Status: OK
# CUBLAS_WORKSPACE_CONFIG=:4096:8 python cnn_pre.py

import sys
import torch

import numpy as np

from networks import CNN_Wrapper
from utils import read_json
from tabulate import tabulate


def CNN(model_name, config, Wrapper, num_exps):
    print('Evaluation metric: {}'.format(config['loss_metric']))
    reports = [[],[],[],[]]
    for exp_idx in range(num_exps):
        net = Wrapper(config, model_name, exp_idx)
        # cnn.load('./checkpoint_dir/{}_exp{}/'.format('cnn_mri_pre', 0), fixed=False)
        net.train(epochs = config['train_epochs'], training_prints=2)
        reports[0].append(net.test_b(out=True,key='test'))
        reports[1].append(net.test_b(out=True,key='train'))
        reports[2].append(net.test_b(out=True,key='valid'))
    ress = [[model_name+'_Test'], [model_name+'_Train'], [model_name+'_Valid']]

    for rep, res in zip(reports, ress):
        accs = [r['accuracy'] for r in rep]
        res += ['%.3f+-%.3f' % (np.mean(accs), np.std(accs))]
        pms = [r['macro avg']['precision'] for r in rep]
        pws = [r['weighted avg']['precision'] for r in rep]
        res += ['%.3f+-%.3f' % (np.mean(pms), np.std(pms))]
        res += ['%.3f+-%.3f' % (np.mean(pws), np.std(pws))]
        rms = [r['macro avg']['recall'] for r in rep]
        rws = [r['weighted avg']['recall'] for r in rep]
        res += ['%.3f+-%.3f' % (np.mean(rms), np.std(rms))]
        res += ['%.3f+-%.3f' % (np.mean(rws), np.std(rws))]
        fms = [r['macro avg']['f1-score'] for r in rep]
        fws = [r['weighted avg']['f1-score'] for r in rep]
        res += ['%.3f+-%.3f' % (np.mean(fms), np.std(fms))]
        res += ['%.3f+-%.3f' % (np.mean(fws), np.std(fws))]

    return ress


def main():
    num_exps = 1
    table = []
    table.append(['Model', 'Accuracy', 'Precision (macro avg)', 'Precision (weighted avg)', 'Recall (macro avg)', 'Recall (weighted avg)', 'F1-score (macro avg)', 'F1-score (weighted avg)'])
    torch.use_deterministic_algorithms(True)
    print('-'*100)
    print('Pretraining CNN classifier for AD status')
    config = read_json('./config.json')['cnn_P']
    model_name = 'CNN_{}_Pre'.format(config['loss_metric'])
    ress = CNN(model_name, config, CNN_Wrapper, num_exps)
    table.append(ress[0])
    table.append(ress[1])
    table.append(ress[2])
    print('-'*100)
    print(tabulate(table, headers='firstrow'))
    print('-'*100)
    print('Completed')


if __name__ == "__main__":
    main()
