# main file for CNN classifier evaluation
# Updated: 2/27/2021
# Status: OK
# CUBLAS_WORKSPACE_CONFIG=:4096:8 python cnn_eval.py

import sys, os
import torch

import numpy as np

from networks import CNN_Wrapper
from utils import read_json
from tabulate import tabulate

def CNN_eval(model_name, config, Wrapper, num_exps):
    print('Evaluation metric: {}'.format(config['loss_metric']))
    reports_all = []
    for exp_idx in range(num_exps):
        net = Wrapper(config, model_name, exp_idx)
        reports, names = net.test(out=True)
        reports_all.append(reports)
    accs = [[] for i in range(10)] #5 types - T Z G G1 G2; 2 datasets - ADNI, NACC
    pws = [[] for i in range(10)]
    rws = [[] for i in range(10)]
    fws = [[] for i in range(10)]
    ress = [[model_name], [model_name+'_E']]
    for rep in reports_all:
        for id, r in enumerate(rep):
            accs[id] += [r['accuracy']]
            pws[id] += [r['weighted avg']['precision']]
            rws[id] += [r['weighted avg']['recall']]
            fws[id] += [r['weighted avg']['f1-score']]
    table = [['Model', 'MRI Type', 'Accuracy', 'Precision', 'Recall', 'F1-score']]
    for id in range(10):
        res = []
        res += [model_name]
        res += [names[id]]
        res += ['%.3f+-%.3f' % (np.mean(accs[id]), np.std(accs[id]))]
        res += ['%.3f+-%.3f' % (np.mean(pws[id]), np.std(pws[id]))]
        res += ['%.3f+-%.3f' % (np.mean(rws[id]), np.std(rws[id]))]
        res += ['%.3f+-%.3f' % (np.mean(fws[id]), np.std(fws[id]))]
        table += [res]
    with open('report.txt', 'a') as f:
        f.write(tabulate(table, headers='firstrow'))

def main():
    num_exps = 5
    torch.use_deterministic_algorithms(True)
    data_root = '/data1/RGAN_Data/'
    print('Legend: T (Original ADNI), Z (Sliced ADNI), [otherwise] (Generated ADNI)')
    datasets = ['T', 'Z', 'G', 'CG_1', 'CG_2']
    if os.path.exists('report.txt.'):
        os.remove('report.txt.')
    for d in datasets:
        print('-'*100)
        print('Evaluating CNN classifier; Training Dataset: '+d)
        config = read_json('./config.json')['cnn_E']
        config['data_dir'] = data_root+d+'/'
        model_name = 'CNN_{}_'.format(config['loss_metric'])+d
        CNN_eval(model_name, config, CNN_Wrapper, num_exps)
        print('-'*100)
    print('-'*100)
    print('Completed')


if __name__ == "__main__":
    main()
