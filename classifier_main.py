# main file for CNN classifier
# Updated: 9/16/2023
# Status: OK
# CUBLAS_WORKSPACE_CONFIG=:4096:8 python classifier_main.py

# results in current setting:
# ----------------------------------------------------------------------------------------------------
# Model                Accuracy      Precision (weighted avg)    Recall (weighted avg)    F1-score (weighted avg)
# -------------------  ------------  --------------------------  -----------------------  -------------------------
# CNN_Standard_I       0.766+-0.055  0.759+-0.062                0.766+-0.055             0.743+-0.066
# CNN_Standard_I_E     0.669+-0.014  0.683+-0.008                0.669+-0.014             0.663+-0.018
# CNN_Standard_T       0.794+-0.033  0.795+-0.032                0.794+-0.033             0.785+-0.036
# CNN_Standard_T_E     0.675+-0.006  0.683+-0.004                0.675+-0.006             0.671+-0.007
# CNN_Standard_Z       0.703+-0.064  0.681+-0.082                0.703+-0.064             0.665+-0.080
# CNN_Standard_Z_E     0.597+-0.021  0.624+-0.021                0.597+-0.021             0.573+-0.028
# CNN_Standard_G       0.709+-0.069  0.711+-0.084                0.709+-0.069             0.653+-0.079
# CNN_Standard_G_E     0.589+-0.025  0.625+-0.021                0.589+-0.025             0.556+-0.039
# CNN_Standard_CG_1    0.709+-0.042  0.697+-0.062                0.709+-0.042             0.694+-0.060
# CNN_Standard_CG_1_E  0.640+-0.017  0.650+-0.014                0.640+-0.017             0.634+-0.019
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------gn
# Model                Accuracy      Precision (weighted avg)    Recall (weighted avg)    F1-score (weighted avg)
# -------------------  ------------  --------------------------  -----------------------  -------------------------
# CNN_Standard_T       0.743+-0.057  0.751+-0.067                0.743+-0.057             0.732+-0.060
# CNN_Standard_T_E     0.647+-0.010  0.656+-0.006                0.647+-0.010             0.642+-0.013
# CNN_Standard_Z       0.651+-0.042  0.660+-0.036                0.651+-0.042             0.636+-0.049
# CNN_Standard_Z_E     0.566+-0.039  0.559+-0.057                0.566+-0.039             0.540+-0.089
# CNN_Standard_G       0.674+-0.084  0.641+-0.131                0.674+-0.084             0.641+-0.098
# CNN_Standard_G_E     0.567+-0.039  0.578+-0.043                0.567+-0.039             0.538+-0.066
# CNN_Standard_CG_1    0.714+-0.097  0.719+-0.118                0.714+-0.097             0.706+-0.100
# CNN_Standard_CG_1_E  0.608+-0.051  0.612+-0.047                0.608+-0.051             0.599+-0.063
# ----------------------------------------------------------------------------------------------------
# T test MCC  0.4117+/-0.1333
# T ext MCC  0.3032+/-0.0154
# Z test MCC  0.2061+/-0.1117
# Z ext MCC  0.1284+/-0.0895
# G test MCC  0.1844+/-0.2395
# G ext MCC  0.1448+/-0.0810
# CG_1 test MCC  0.3402+/-0.2577
# CG_1 ext MCC  0.2207+/-0.0982


import sys
import torch

import numpy as np

from networks import CNN_Wrapper
from utils import read_json
from tabulate import tabulate


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
        # net.test_b(key='test')
        # net.test_b(key='ext')
        # net.test_b(key='train')
        # net.test_b(key='valid')
        # sys.exit()
        reports[0].append(net.test_b(out=True,key='test'))
        reports[1].append(net.test_b(out=True,key='ext'))
        reports[2].append(net.test_b(out=True,key='train'))
        reports[3].append(net.test_b(out=True,key='valid'))
        net.shap()
    ress = [[model_name], [model_name+'_E'], [model_name+'_Train'], [model_name+'_Valid']]
    # print(ress)
    # print(reports)

    for rep, res in zip(reports, ress):
        # break
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
    num_exps = 5
    table = []
    table.append(['Model', 'Accuracy', 'Precision (weighted avg)', 'Recall (weighted avg)', 'F1-score (weighted avg)'])
    # table.append(['Model', 'Accuracy', 'Precision (macro avg)', 'Precision (weighted avg)', 'Recall (macro avg)', 'Recall (weighted avg)', 'F1-score (macro avg)', 'F1-score (weighted avg)'])
    torch.use_deterministic_algorithms(True)
    data_root = '/data1/RGAN_Data/'
    # datasets = ['I', 'T', 'Z', 'G', 'CG_1', 'CG_2']
    # datasets = ['I', 'T', 'Z', 'G', 'CG_1']
    datasets = ['T', 'Z', 'G', 'CG_1']
    # datasets = ['G', 'CG_1']
    # datasets = ['G']
    # datasets = ['I']
    # datasets = ['CG_1']
    # datasets = ['T', 'Z'] 
    
    # datasets = ['CG_2']
    print('Explanation: T (Original ADNI), Z (Sliced ADNI), [otherwise] (Generated ADNI)')
    train = False
    # train = True
    for d in datasets:
        print('-'*100)
        print('Running CNN & MLP classifiers for AD status; Dataset: '+d)
        config = read_json('./config.json')['cnn_E']
        config['data_dir'] = data_root+d+'/'
        # model_name = 'CNN_'+d
        model_name = 'CNN_{}_'.format(config['loss_metric'])+d
        ress = CNN(model_name, config, CNN_Wrapper, num_exps, train)
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
