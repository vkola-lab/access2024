# dataloader for RGAN
# Created: 6/16/2021
# Status: ok

import random
import glob
import os, sys

import numpy as np
import nibabel as nib

from torch.utils.data import Dataset
from utils import read_csv_sp as read_csv, read_csv_cox_ext as read_csv_ext, rescale, read_csv_pre

SCALE = 1 #rescale to 0~2.5

class B_Data(Dataset):
    #Brain data
    def __init__(self, data_dir, stage, ratio=(0.8, 0.1, 0.1), seed=1000, step_size=10, external=False, Pre=False):
        random.seed(seed)

        self.stage = stage
        self.data_dir = data_dir
        self.step_size = step_size

        # self.data_list = glob.glob(data_dir + 'coregistered*nii*')
        self.data_list = glob.glob(data_dir + '*nii*')

        csvname = './csvs/merged_dataframe_cox_noqc_pruned_final.csv'
        if external:
            csvname = './csvs/merged_dataframe_cox_test_pruned_final.csv'
        elif Pre:
            csvname = './csvs/merged_dataframe_unused_cox_pruned.csv'
        csvname = os.path.expanduser(csvname)
        if external:
            fileIDs, time_hit = read_csv_ext(csvname) #training file
        elif Pre:
            fileIDs, time_hit = read_csv_pre(csvname) #training file
        else:
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
        # print(l)
        # print('pmci', self.time_hit.count(1))
        # sys.exit()
        split1 = int(l*ratio[0])
        split2 = int(l*(ratio[0]+ratio[1]))
        idxs = list(range(l))
        random.shuffle(idxs)
        if 'train' in stage:
            self.index_list = idxs[:split1]
            # print(len(self.index_list))
        elif 'valid' in stage:
            self.index_list = idxs[split1:split2]
            # print(len(self.index_list))
        elif 'test' in stage:
            self.index_list = idxs[split2:]
            # print(len(self.index_list))
        elif 'all' in stage:
            self.index_list = idxs
            # print(len(self.index_list))
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

class B_IQ_Data(Dataset):
    #Brain data
    def __init__(self, data_dir, stage, ratio=(0.8, 0.1, 0.1), seed=1000, step_size=10, external=False):
        random.seed(seed)

        self.stage = stage
        self.names=['T', 'Z', 'G', 'CG_1', 'CG_2']
        if external:
            self.names = [n+'_E' for n in self.names]
        self.names = [n+'/' for n in self.names]
        self.step_size = step_size

        # self.data_list = glob.glob(data_dir + 'coregistered*nii*')
        self.data_dir = data_dir
        self.data_list = glob.glob(data_dir+self.names[0] + '*nii*')

        csvname = './csvs/merged_dataframe_cox_noqc_pruned_final.csv'
        if external:
            csvname = './csvs/merged_dataframe_cox_test_pruned_final.csv'
        csvname = os.path.expanduser(csvname)
        if external:
            fileIDs, time_hit = read_csv_ext(csvname) #training file
        else:
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

        datas = []
        for n in self.names:
            filename = self.data_list[idx].replace(self.names[0],n)

            data = nib.load(filename).get_fdata().astype(np.float32)
            data[data != data] = 0
            SCALE=0
            if SCALE:
                data = rescale(data, (0, 2.5))
                if 0:
                    data = rescale(data, (0, 99))
                    data = data.astype(np.int)
            # data = np.expand_dims(data, axis=0)
            datas.append(data)
        # print(len(datas))
        # print(datas[0].shape)
        # sys.exit()
        return datas


        # return data, obs, hit

    def get_sample_weights(self):
        num_classes = len(set(self.time_hit))
        counts = [self.time_hit.count(i) for i in range(num_classes)]
        count = len(self.time_hit)
        weights = [count / counts[i] for i in self.time_hit]
        class_weights = [count/c for c in counts]
        return weights, class_weights


if __name__ == "__main__":
    Data_dir_NACC = "/data2/MRI_PET_DATA/processed_images_final_cox_test/brain_stripped_cox_test/"
    Data_dir_ADNI = "/data2/MRI_PET_DATA/processed_images_final_unused_cox/brain_stripped_unused_cox/"
    external_data = B_Data(Data_dir_ADNI, 'all', Pre=True)
    print(len(external_data))
    for _ in external_data:
        print()
        sys.exit()
