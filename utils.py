# util functions for RGAN
# Created: 6/16/2021
# Status: ok

import json
import csv
import os

import numpy as np

def read_json(filename):
    with open(filename) as buf:
        return json.loads(buf.read())

def read_csv_cox(filename):
    with open(filename, 'r') as f:
        # reader = csv.reader(f)
        # your_list = list(reader)
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
                    status += [0]
    fileIDs = ['0'*(4-len(f))+f for f in fileIDs]
    # print('0', status.count(0))
    # print('1', status.count(1))
    return fileIDs, status

def read_csv_sp(filename):
    #only considers smci and pmci cases
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        fileIDs, status = [], []
        smci = 0
        pmci = 0
        for r in reader:
            temp = r['TIMES']
            if len(temp) == 0:
                continue
            else:
                fileIDs += [str(int(float(r['RID'])))]
                time = int(float(r['TIMES']))
                # 0 for smci, 1 for pmci (progress within 36 months)
                status += [int(time<36)]
    fileIDs = ['0'*(4-len(f))+f for f in fileIDs]
    # print('smci', status.count(0))
    # print('pmci', status.count(1))
    return fileIDs, status

def rescale(array, tup):
    m = np.min(array)
    if m < 0:
        array += -m
    a = np.max(array)-np.min(array)
    t = tup[1] - tup[0]
    return array * t / a

if __name__ == "__main__":
    csvname = '~/3d_rgan/source/merged_dataframe_cox_noqc_pruned_final.csv'
    csvname = os.path.expanduser(csvname)
    read_csv_cox(csvname)
    #37 smci, 117 pmci
