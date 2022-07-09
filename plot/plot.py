from utils_stat import get_roc_info, get_pr_info, load_neurologist_data, calc_neurologist_statistics, read_raw_score
import matplotlib.pyplot as plt
from utils_plot import plot_curve, plot_legend, plot_neorologist
from time import time
from matrix_stat import confusion_matrix, stat_metric
import collections
from tabulate import tabulate
import os
from matplotlib.patches import Rectangle
from scipy import stats
from sklearn.metrics import classification_report, roc_curve, RocCurveDisplay
import csv
import numpy as np
import sys

def p_val(o, g):
    t, p = stats.ttest_ind(o, g, equal_var = False)
    # print(o, g, p)
    return p

def plot_legend(axes, crv_lgd_hdl, crv_info, set1, set2):
    m_name = list(crv_lgd_hdl.keys())
    ds_name = list(crv_lgd_hdl[m_name[0]].keys())

    hdl = collections.defaultdict(list)
    val = collections.defaultdict(list)

    for ds in ds_name:
        for m in m_name:
            # print(m, ds)
            hdl[ds].append(crv_lgd_hdl[m][ds])
            val[ds].append('{}: {:.3f}$\pm${:.3f}'.format(m, crv_info[m][ds]['auc_mean'], crv_info[m][ds]['auc_std']))
        extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
        # val[ds].append('p-value: {:.4e}'.format(p_val(set1[ds], set2[ds])))

        axes[ds].legend(hdl[ds]+[extra], val[ds],
                        facecolor='w', prop={"weight":'bold', "size":17},  # frameon=False,
                        bbox_to_anchor=(0.5, 0.5, 0.5, 0.5),
                        loc='lower left')

def roc_plot_perfrom_table(txt_file=None, mode=['T', 'Z', 'G', 'CG_1', 'CG_2']):
    roc_info, pr_info = {}, {}
    aucs, apss = {}, {}
    datas = ['test', 'ext']
    for m in mode:
        roc_info[m], pr_info[m], aucs[m], apss[m] = {}, {}, {}, {}
        for ds in datas:
            Scores, Labels = [], []
            x = 3
            # csvname = os.path.expanduser(csvname)
            for exp_idx in range(x):
                labels, scores = read_raw_score('../checkpoint_dir/CNN_Standard_{}/raw_score_{}_{}.txt'.format(m, ds, exp_idx))
                Scores.append(scores)
                Labels.append(labels)

            roc_info[m][ds], aucs[m][ds] = get_roc_info(Labels, Scores)
            pr_info[m][ds], apss[m][ds] = get_pr_info(Labels, Scores)
            
    plt.style.use('fivethirtyeight')
    plt.rcParams['axes.facecolor'] = 'w'
    plt.rcParams['figure.facecolor'] = 'w'
    plt.rcParams['savefig.facecolor'] = 'w'

    # roc plot
    fig, axes_ = plt.subplots(1, 2, figsize=[18, 6], dpi=100)
    axes = dict(zip(datas, axes_))
    lines = ['-', '--', '-.', ':', ' ']
    hdl_crv = {m:{} for m in mode}
    for i, ds in enumerate(datas):
        title = ds.replace('ext', 'NACC')
        i += 1
        for j, m in enumerate(mode):
            hdl_crv[m][ds] = plot_curve(curve='roc', **roc_info[m][ds], ax=axes[ds],
                                    **{'color': 'C{}'.format(i+2*j), 'hatch': '//////', 'alpha': .8, 'line': lines[j],
                                        'title': title})

    plot_legend(axes=axes, crv_lgd_hdl=hdl_crv, crv_info=roc_info, set1=aucs[mode[0]], set2=aucs[mode[1]])
    fig.savefig('./roc.png', dpi=300)

    # pr plot
    fig, axes_ = plt.subplots(1, 2, figsize=[18, 6], dpi=100)
    axes = dict(zip(datas, axes_))
    hdl_crv = {m: {} for m in mode}
    for i, ds in enumerate(datas):
        title = ds.replace('ext', 'NACC')
        i += 1
        for j, m in enumerate(mode):
            hdl_crv[m][ds] = plot_curve(curve='pr', **pr_info[m][ds], ax=axes[ds],
                                    **{'color': 'C{}'.format(i+2*j), 'hatch': '//////', 'alpha': .8, 'line': lines[j],
                                        'title': title})

    plot_legend(axes=axes, crv_lgd_hdl=hdl_crv, crv_info=pr_info, set1=apss[mode[0]], set2=apss[mode[1]])
    fig.savefig('./pr.png', dpi=300)

def ci(mean, std, N):
    return [mean-1.96*std/np.sqrt(N), mean+1.96*std/np.sqrt(N)]

if __name__ == "__main__":
    roc_plot_perfrom_table()
