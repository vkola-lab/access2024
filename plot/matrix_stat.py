from utils_stat import (
    get_roc_info,
    get_pr_info,
    calc_neurologist_statistics,
    read_raw_score,
)
import matplotlib.pyplot as plt
from utils_plot import plot_curve, plot_legend, plot_neorologist
from time import time
import numpy as np


def confusion_matrix(labels, scores):
    matrix = [[0, 0], [0, 0]]
    for label, pred in zip(labels, scores):
        if pred < 0.5:
            if label == 0:
                matrix[0][0] += 1
            if label == 1:
                matrix[0][1] += 1
        else:
            if label == 0:
                matrix[1][0] += 1
            if label == 1:
                matrix[1][1] += 1
    return matrix


def get_metrics(matrix):
    TP, FP, TN, FN = matrix[1][1], matrix[1][0], matrix[0][0], matrix[0][1]
    TP, FP, TN, FN = float(TP), float(FP), float(TN), float(FN)
    # print(TP,TN,FP,FN)
    try:
        ACCU = (TP + TN) / (TP + TN + FP + FN)
    except:
        print(matrix)
        sys.exit()
    Sens = TP / (TP + FN + 0.0000001)
    Spec = TN / (TN + FP + 0.0000001)
    F1 = 2 * TP / (2 * TP + FP + FN)
    MCC = (TP * TN - FP * FN) / (
        (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) + 0.00000001
    ) ** 0.5
    return ACCU, Sens, Spec, F1, MCC


def stat_metric(matrices):
    Accu, Sens, Spec, F1, MCC = [], [], [], [], []
    for matrix in matrices:
        accu, sens, spec, f1, mcc = get_metrics(matrix)
        Accu.append(accu)
        Sens.append(sens)
        Spec.append(spec)
        F1.append(f1)
        MCC.append(mcc)
    # print('Accu {0:.4f}+/-{1:.4f}'.format(float(np.mean(Accu)), float(np.std(Accu))))
    # print('Sens {0:.4f}+/-{1:.4f}'.format(float(np.mean(Sens)), float(np.std(Sens))))
    # print('Spec {0:.4f}+/-{1:.4f}'.format(float(np.mean(Spec)), float(np.std(Spec))))
    # print('F1   {0:.4f}+/-{1:.4f}'.format(float(np.mean(F1)),   float(np.std(F1))))
    MCC_text = "MCC  {0:.4f}+/-{1:.4f}".format(float(np.mean(MCC)), float(np.std(MCC)))
    group = (
        float(np.mean(Accu)),
        float(np.std(Accu)),
        float(np.mean(Sens)),
        float(np.std(Sens)),
        float(np.mean(Spec)),
        float(np.std(Spec)),
        float(np.mean(F1)),
        float(np.std(F1)),
        float(np.mean(MCC)),
        float(np.std(MCC)),
    )
    return group, MCC_text


def compute_mlp_matrix_stat():
    for dataset in (
        "NACC",
        "ADNI",
    ):
        if dataset == "NACC":
            stage = "all"
        elif dataset == "ADNI":
            stage = "test"
        else:
            raise NotImplementedError
        Matrix = []
        for exp in range(5):
            labels, scores = read_raw_score(
                f"../checkpoint_dir/mlp_bce_{exp}_exp{exp}/raw_score_{dataset}_{exp}_{stage}.txt"
            )
            Matrix.append(confusion_matrix(labels, scores))
        print(stat_metric(Matrix))


if __name__ == "__main__":
    # names = ['T', 'Z', 'G', 'CG_1']
    # datas = ['test', 'ext']
    # table = []
    # table.append(['Model', 'Accuracy', 'Precision (weighted avg)', 'Recall (weighted avg)', 'F1-score (weighted avg)'])
    # for n in names:
    #     for d in datas:
    #         Matrix = []
    #         for i in range(5):
    #             folder = 'CNN_Standard_'+n+str(i)
    #             labels, scores = read_raw_score('../checkpoint_dir/'+folder+'/raw_score_{}_{}.txt'.format(d, i))
    #             Matrix.append(confusion_matrix(labels, scores))
    #         _, txt = stat_metric(Matrix)
    #         print(n, d, txt)
    compute_mlp_matrix_stat()
