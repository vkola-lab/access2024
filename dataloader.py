# dataloader for RGAN
# Created: 6/16/2021
# Status: ok

import random
import glob
import os, sys

import numpy as np
import nibabel as nib

from torch.utils.data import Dataset
from utils import read_csv_cox as read_csv, rescale

SCALE = 1 #rescale to 0~2.5

class B_Data(Dataset):
    #Brain data
    def __init__(self, data_dir, stage, ratio=(0.6, 0.2, 0.2), seed=1000, step_size=10):
        random.seed(seed)

        self.stage = stage
        self.data_dir = data_dir
        self.step_size = step_size

        # self.data_list = glob.glob(data_dir + 'coregistered*nii*')
        self.data_list = glob.glob(data_dir + '*nii*')

        # csvname = '~/mri-pet/metadata/data_processed/merged_dataframe_cox_pruned_final.csv'
        csvname = './merged_dataframe_cox_noqc_pruned_final.csv'
        csvname = os.path.expanduser(csvname)
        fileIDs, time_hit = read_csv(csvname) #training file

        tmp_f = []
        tmp_h = []
        tmp_d = []
        for d in self.data_list:
            for f, h in zip(fileIDs, time_hit):
                fname = os.path.basename(d)
                if f in fname:
                    tmp_f.append(f)
                    tmp_h.append(h)
                    tmp_d.append(d)
                    break
        self.data_list = tmp_d
        self.time_hit  = tmp_h
        self.fileIDs   = tmp_f #Note: this only for csv generation not used for data retrival

        # print(len(tmp_f))
        l = len(self.data_list)
        split1 = int(l*ratio[0])
        split2 = int(l*(ratio[0]+ratio[1]))
        idxs = list(range(len(fileIDs)))
        random.shuffle(idxs)
        if 'train' in stage:
            self.index_list = idxs[:split1]
        elif 'valid' in stage:
            self.index_list = idxs[split1:split2]
        elif 'test' in stage:
            self.index_list = idxs[split2:]
        elif 'all' in stage:
            self.index_list = idxs
        else:
            raise Exception('Unexpected Stage for Vit_Data!')
        # print(len(self.index_list))
        # print((self.fileIDs[:10]))
        # sys.exit()

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):
        idx = self.index_list[idx]
        hit = self.time_hit[idx]

        data = nib.load(self.data_list[idx]).get_fdata().astype(np.float32)
        data[data != data] = 0
        if SCALE:
            data = rescale(data, (0, 2.5))
            if 0:
                data = rescale(data, (0, 99))
                data = data.astype(np.int)
        data = np.expand_dims(data, axis=0)

        g_data = data[:,::self.step_size]
        # print('dataloader', data.shape)
        # print('dataloader', g_data.shape)

        # sys.exit()
        return g_data, data, self.data_list[idx], hit


        # return data, obs, hit

    def get_sample_weights(self):
        num_classes = len(set(self.time_hit))
        counts = [self.time_hit.count(i) for i in range(num_classes)]
        count = len(self.time_hit)
        weights = [count / counts[i] for i in self.time_hit]
        class_weights = [count/c for c in counts]
        return weights, class_weights
