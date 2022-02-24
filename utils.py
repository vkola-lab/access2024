# util functions for RGAN
# Created: 6/16/2021
# Status: ok

import json
import csv
import os, sys
import matlab.engine
import collections

import numpy as np

from skimage import img_as_float
from scipy.interpolate import interp1d
from scipy import stats
from skimage.metrics import structural_similarity as ssim


def read_json(filename):
    with open(filename) as buf:
        return json.loads(buf.read())

def write_raw_score(f, preds, labels):
    # preds = preds.data.cpu().numpy()
    # labels = labels.data.cpu().numpy()
    for index, pred in enumerate(preds):
        pred = pred.data.cpu().numpy()
        label = str(labels[index].data.cpu().numpy())
        pred = "__".join(map(str, [1-pred, pred]))
        f.write(pred + '__' + label + '\n')

def read_csv_cox(filename):
    #Note: this is MCI vs AD; MCI is considered as long as there is one MCI in the labels (if AD not appeared)
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        fileIDs, status = [], []
        for r in reader:
            temp = r['PROGRESSES']
            if len(temp) == 0:
                continue
            else:
                fileIDs += [str(int(float(r['RID'])))]
                try:
                    status += [int(float(r['PROGRESSES']))]
                except:
                    fileIDs.pop(-1)
    fileIDs = ['0'*(4-len(f))+f for f in fileIDs]
    # print('0', status.count(0))
    # print('1', status.count(1))
    # print('0:0+1', status.count(0)/len(status))
    return fileIDs, status

def read_csv_cox_ext(filename):
    #Note: this is MCI vs AD; MCI is considered as long as there is one MCI in the labels (if AD not appeared)
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        fileIDs, status = [], []
        ttt = 0
        for r in reader:
            temp = r['PROGRESSES']
            if len(temp) == 0:
                continue
            # 318 labels unavailable for smci/pmci
            time = int(float(r['TIMES']))
            if float(temp) == 0 and time <= 36:
                ttt += 1
            else:
                fileIDs += [str(r['RID'])]
                status += [int(time<=36)]
    # print(ttt)
    # print('ext smci', status.count(0))
    # not progressed time < 36
    # print('ext pmci', status.count(1))
    # print('0:0+1', status.count(0)/len(status))
    return fileIDs, status

def read_csv_sp(filename):
    #only considers smci and pmci cases
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        fileIDs, status = [], []
        ttt = 0
        smci = 0
        pmci = 0
        for r in reader:
            temp = r['PROGRESSES']
            if len(temp) == 0:
                continue
            # 200 labels unavailable for smci/pmci
            time = int(float(r['TIMES']))
            if float(temp) == 0 and time <= 36:
                ttt += 1
            else:
                fileIDs += [str(int(float(r['RID'])))]
                # 0 for smci, 1 for pmci (progress within 36 months)
                status += [int(time<=36)]
    fileIDs = ['0'*(4-len(f))+f for f in fileIDs]
    # print(ttt)
    # print('smci', status.count(0))
    # print('pmci', status.count(1))
    return fileIDs, status

def read_csv_pre(filename):
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        fileIDs, status = [], []
        for r in reader:
            l = str(r['DX'])
            if l == 'AD':
                l = 1
            elif l == 'MCI':
                l = 0
                # continue
            elif l == 'NL':
                l = 0
            else:
                continue
            status += [l]
            fileIDs += [str(r['FILE_CODE'])]
    # print('smci', status.count(0))
    # print('pmci', status.count(1))
    # print(len(fileIDs), len(status))
    return fileIDs, status

def rescale(array, tup):
    m = np.min(array)
    if m < 0:
        array += -m
    a = np.max(array)-np.min(array)
    t = tup[1] - tup[0]
    return array * t / a

def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)

def SNR(tensor):
    # return the signal to noise ratio
    for slice_idx in [80]:
        img = tensor[:, :, slice_idx]
        m = interp1d([np.min(img),np.max(img)],[0,255])
        img = m(img)
        val = signaltonoise(img, axis=None)
    return float(val)

def CNR(tensor):
    # return the signal to noise ratio
    for slice_idx in [80]:
        img = tensor[:, :, slice_idx] #shape 121, 145, (121)
        # print(img.shape)
        m = interp1d([np.min(img),np.max(img)],[0,255])
        img = m(img)
        roi1, roi2 = img[90:120, 80:110], img[60:90, 50:80]
        return np.abs(np.mean(roi1) - np.mean(roi2)) / np.sqrt(np.square(np.std(roi1))+np.square(np.std(roi2)))

def SSIM(tensor1, tensor2, zoom=False):
    ssim_list = []
    for slice_idx in [60]:
        if zoom:
            side_a = slice_idx
            side_b = slice_idx+30
            img1, img2 = tensor1[side_a:side_b, side_a:side_b, 60], tensor2[side_a:side_b, side_a:side_b, 60]
        else:
            img1, img2 = tensor1[:, :, slice_idx], tensor2[:, :, slice_idx]
        img1 = img_as_float(img1)
        img2 = img_as_float(img2)
        ssim_val = ssim(img1, img2)
        if ssim_val != ssim_val:
            print('\n\n Error @ SSIM')
            sys.exit()
        ssim_list.append(ssim_val)
    ssim_avg = sum(ssim_list) / len(ssim_list)
    return ssim_avg

def immse(tensor1, tensor2, zoom, eng):
    vals = []
    for slice_idx in [50, 80, 110]:
        if zoom:
            side_a = slice_idx
            side_b = slice_idx+60
            img1, img2 = tensor1[side_a:side_b, side_a:side_b, 105], tensor2[side_a:side_b, side_a:side_b, 105]
        else:
            img1, img2 = tensor1[slice_idx, :, :], tensor2[slice_idx, :, :]
        img1, img2 = matlab.double(img1.tolist()), matlab.double(img2.tolist())
        val = eng.immse(img1, img2)
        vals.append(val)
    val_avg = sum(vals) / len(vals)
    return val_avg

def psnr(tensor1, tensor2, zoom, eng):
    #all are single slice!
    vals = []
    for slice_idx in [50, 80, 110]:
        if zoom:
            side_a = slice_idx
            side_b = slice_idx+60
            img1, img2 = tensor1[side_a:side_b, side_a:side_b, 105], tensor2[side_a:side_b, side_a:side_b, 105]
        else:
            img1, img2 = tensor1[slice_idx, :, :], tensor2[slice_idx, :, :]
        img1, img2 = matlab.double(img1.tolist()), matlab.double(img2.tolist())
        val = eng.psnr(img1, img2)
        vals.append(val)
    val_avg = sum(vals) / len(vals)
    return val_avg

def iqa_tensor(tensor, eng, metric, filename='', target=''):
    # if not os.path.isdir(target):
    #     os.mkdir(target)
    # in_size = 121*145*121
    out = []
    if metric == 'brisque':
        func = eng.brisque
    elif metric == 'niqe':
        func = eng.niqe
    elif metric == 'piqe':
        func = eng.piqe
    elif metric == 'CNR':
        return CNR(tensor)
    elif metric == 'SNR':
        return SNR(tensor)


    for side in range(len(tensor.shape)):
        vals = []
        for slice_idx in [70]:
            if side == 0:
                img = tensor[slice_idx, :, :]
            elif side == 1:
                img = tensor[:, slice_idx, :]
            else:
                img = tensor[:, :, slice_idx]
            img = matlab.double(img.tolist())
            vals += [func(img)]
        out += vals
    val_avg = sum(out) / len(out)
    #np.save(target+filename+'$'+metric, out)
    # return np.asarray(out)
    return val_avg

def p_val(o, g):
    t, p = stats.ttest_ind(o, g, equal_var=True)
    return p

if __name__ == "__main__":
    csvname = './csvs/merged_dataframe_cox_noqc_pruned_final.csv'
    csvname = os.path.expanduser(csvname)
    # read_csv_cox(csvname)
    read_csv_sp(csvname)
    csvname = './csvs/merged_dataframe_cox_test_pruned_final.csv' #(NACC)
    csvname = os.path.expanduser(csvname)
    read_csv_cox_ext(csvname)
    #37 smci, 117 pmci
    csvname = './csvs/merged_dataframe_unused_cox_pruned.csv' #(ADNI-pre)
    csvname = os.path.expanduser(csvname)
    read_csv_pre(csvname)
