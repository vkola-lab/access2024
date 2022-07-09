# main file for CNN classifier
# Updated: 6/8/2022
# Status: OK
# CUBLAS_WORKSPACE_CONFIG=:4096:8 python cnn_main.py

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
        # net.visualize()
        # print('loading model...')
        net.load('./checkpoint_dir/CNN_Standard_Pre0/', fixed=False)
        net.train(epochs = config['train_epochs'], training_prints=2)
        net.visualize(prefix=model_name)
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
        # fms = [r['macro avg']['f1-score'] for r in rep]
        # res += ['%.3f+-%.3f' % (np.mean(fms), np.std(fms))]

    return ress

def main():
    num_exps = 2
    table = []
    table.append(['Model', 'Accuracy', 'Precision (weighted avg)', 'Recall (weighted avg)', 'F1-score (weighted avg)'])
    # table.append(['Model', 'Accuracy', 'Precision (macro avg)', 'Precision (weighted avg)', 'Recall (macro avg)', 'Recall (weighted avg)', 'F1-score (macro avg)', 'F1-score (weighted avg)'])
    torch.use_deterministic_algorithms(True)
    data_root = '/data1/RGAN_Data/'
    # datasets = ['T', 'Z', 'G', 'CG_1', 'CG_2']
    # datasets = ['T', 'Z', 'G', 'CG_1']
    datasets = ['T', 'Z', 'CG_1'] 
    
    # datasets = ['CG_2']
    print('Explanation: T (Original ADNI), Z (Sliced ADNI), [otherwise] (Generated ADNI)')
    for d in datasets:
        print('-'*100)
        print('Running CNN & MLP classifiers for AD status; Dataset: '+d)
        config = read_json('./config.json')['cnn_E']
        config['data_dir'] = data_root+d+'/'
        model_name = 'CNN_{}_'.format(config['loss_metric'])+d
        ress = CNN(model_name, config, CNN_Wrapper, num_exps)
        table.append(ress[0])
        table.append(ress[1])
        # table.append(ress[2])
        # table.append(ress[3])
        
        print('-'*100)
    print(tabulate(table, headers='firstrow'))
    print('-'*100)
    print('Completed')


if __name__ == "__main__":
    main()
