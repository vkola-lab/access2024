# dataloader for RGAN
# Created: 6/16/2021
# Status: ok

import copy
import glob
import json
import os
import random
import sys
# from typing import Literal

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from utils import read_csv_cox_ext as read_csv_ext
from utils import read_csv_demog_ed_apoe, read_csv_pre
from utils import read_csv_sp as read_csv
from utils import read_json, rescale

SCALE = 1  # rescale to 0~2.5


class B_Data(Dataset):
    # Brain data
    def __init__(
        self,
        data_dir,
        stage,
        ratio=(0.8, 0.1, 0.1),
        seed=1000,
        step_size=10,
        external=False,
        Pre=False,
    ):
        random.seed(seed)

        self.stage = stage
        self.data_dir = data_dir
        self.step_size = step_size

        # self.data_list = glob.glob(data_dir + 'coregistered*nii*')
        self.data_list = glob.glob(data_dir + "*nii*")

        csvname = "./csvs/merged_dataframe_cox_noqc_pruned_final.csv"
        if external:
            csvname = "./csvs/merged_dataframe_cox_test_pruned_final.csv"
        elif Pre:
            csvname = "./csvs/merged_dataframe_unused_cox_pruned.csv"
        csvname = os.path.expanduser(csvname)
        if external:
            fileIDs, time_hit = read_csv_ext(csvname)  # training file
        elif Pre:
            fileIDs, time_hit = read_csv_pre(csvname)  # training file
        else:
            fileIDs, time_hit = read_csv(csvname)  # training file

        tmp_f = []
        tmp_h = []
        tmp_d = []

        if 1:
            if '_E' in self.data_dir:
                filename = './rids/' + 'NACC/' + 'glob' + '{}.txt'.format(seed)
            else:
                filename = './rids/' + 'ADNI/' + 'glob' + '{}.txt'.format(seed)
            with open(filename, 'a') as f:
                f.write(str(self.data_list)+'\n')
            if '_E' in self.data_dir:
                filename = './rids/' + 'NACC/' + 'fileIDs' + '{}.txt'.format(seed)
            else:
                filename = './rids/' + 'ADNI/' + 'fileIDs' + '{}.txt'.format(seed)
            with open(filename, 'a') as f:
                f.write(str(fileIDs)+'\n')

        for d in self.data_list:
            for f, h in zip(fileIDs, time_hit):
                fname = os.path.basename(d)
                if f in fname:
                    tmp_f.append(f)
                    tmp_h.append(h)
                    tmp_d.append(d)
                    break
        self.data_list = tmp_d
        self.time_hit = tmp_h
        self.fileIDs = tmp_f  # Note: this only for csv generation not used for data retrival
        # print(np.sum(self.time_hit))
        # print(len(tmp_d))
        # print(data_dir)
        # print(len(tmp_f))
        # sys.exit()
        l = len(self.data_list)
        split1 = int(l * ratio[0])
        split2 = int(l * (ratio[0] + ratio[1]))
        idxs = list(range(l))
        random.shuffle(idxs)
        if "train" in stage:
            self.index_list = idxs[:split1]
        elif "valid" in stage:
            self.index_list = idxs[split1:split2]
        elif "test" in stage:
            self.index_list = idxs[split2:]
        elif "all" in stage:
            self.index_list = idxs
            # print(len(self.index_list))
        else:
            raise Exception('Unexpected Stage for Vit_Data!')

        if 0:
            temp_list = np.array(self.data_list)
            out_list = list(temp_list[self.index_list])
            if stage == 'valid':
                print(len(idxs), out_list[:5], len(out_list))
            if '_E' in self.data_dir:
                filename = './rids/' + 'NACC/' + stage + '{}.txt'.format(seed)
            else:
                filename = './rids/' + 'ADNI/' + stage + '{}.txt'.format(seed)
            with open(filename, 'a') as f:
                f.write(str(out_list)+'\n')
        # print(len(self.index_list))
        # print(len(idxs))
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

        g_data = copy.deepcopy(data)
        # randomly remove information

        # g_data[:,::self.step_size] = 0
        indices = list(range(g_data.shape[1]))
        random.shuffle(indices)

        if 0:
            # record removed slices, remember to set generate() input correspondingly.
            filename = './slice_list/NACC/' + os.path.basename(self.data_list[idx]).replace('.nii', '.txt')
            with open(filename, 'w') as f:
                f.write(str(indices[:self.step_size]))
        g_data[:, indices[:self.step_size]] = 0

        # if '2216' in os.path.basename(self.data_list[idx]).replace('.nii', ''):
        #     print(idx)
        #     print(self.index_list)
        #     print(len(self.index_list))
        #     print(indices)
        #     print(os.path.basename(self.data_list[idx]).replace('.nii', ''))
        #     # print('g', g_data[:, :, 80, 80])
        #     sys.exit()

        # return g_data, data, self.data_list[idx], hit
        rid = os.path.basename(self.data_list[idx]).replace('.nii', '')
        return rid, data, self.data_list[idx], hit #use this for shap analysis

    def get_sample_weights(self):
        num_classes = len(set(self.time_hit))
        counts = [self.time_hit.count(i) for i in range(num_classes)]
        count = len(self.time_hit)
        weights = [count / counts[i] for i in self.time_hit]
        class_weights = [count / c for c in counts]
        return weights, class_weights


class B_IQ_Data(Dataset):
    # Brain data
    def __init__(
        self,
        data_dir,
        stage,
        ratio=(0.8, 0.1, 0.1),
        seed=1000,
        step_size=10,
        external=False,
        names=None,#not used for now
    ):
        random.seed(seed)

        self.stage = stage
        self.names = ["T", "Z", "G", "CG_1"]
        if external:
            self.names = [n + "_E" for n in self.names]
        self.names = [n + "/" for n in self.names]
        self.step_size = step_size

        # self.data_list = glob.glob(data_dir + 'coregistered*nii*')
        self.data_dir = data_dir
        self.data_list = glob.glob(data_dir + self.names[0] + "*nii*")

        csvname = "./csvs/merged_dataframe_cox_noqc_pruned_final.csv"
        if external:
            csvname = "./csvs/merged_dataframe_cox_test_pruned_final.csv"
        csvname = os.path.expanduser(csvname)
        if external:
            fileIDs, time_hit = read_csv_ext(csvname)  # training file
        else:
            fileIDs, time_hit = read_csv(csvname)  # training file

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
        self.time_hit = tmp_h
        self.fileIDs = tmp_f  # Note: this only for csv generation not used for data retrival

        # print(len(tmp_f))
        l = len(self.data_list)
        split1 = int(l*ratio[0])
        split2 = int(l*(ratio[0]+ratio[1]))
        idxs = list(range(l))
        random.shuffle(idxs)
        if "train" in stage:
            self.index_list = idxs[:split1]
        elif "valid" in stage:
            self.index_list = idxs[split1:split2]
        elif "test" in stage:
            self.index_list = idxs[split2:]
        elif "all" in stage:
            self.index_list = idxs
        else:
            raise Exception("Unexpected Stage for Vit_Data!")

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):
        idx = self.index_list[idx]

        datas = []
        datas_name = []
        for n in self.names:
            filename = self.data_list[idx].replace(self.names[0], n)

            data = nib.load(filename).get_fdata().astype(np.float32)
            data[data != data] = 0
            SCALE = 0
            if SCALE:
                data = rescale(data, (0, 2.5))
                if 0:
                    data = rescale(data, (0, 99))
                    data = data.astype(np.int)
            # data = np.expand_dims(data, axis=0)
            datas.append(data)
            datas_name.append(filename)
        # print(len(datas))
        # print(datas[0].shape)
        return datas, datas_name


class ParcellationDataBinary(Dataset):
    def __init__(
        self,
        exp_idx,
        stage="train",
        dataset="ADNI",
        ratio=(0.8, 0.1, 0.1),
        seed=1000,
        add_age=False,
        add_mmse=False,
        add_sex=False,
        add_apoe=False,
        add_educ=False,
    ):
        random.seed(seed)
        self.exp_idx = exp_idx
        self.ratio = ratio  # ratios for train/valid/test
        self.stage = stage  # train, validate, or test
        json_props = read_json("./mlp_config.json")
        self.csv_directory = json_props["datadir"]
        self.csvname = self.csv_directory + json_props["metadata_fi"][dataset]
        self.parcellation_file = pd.read_csv(
            self.csv_directory + json_props["parcellation_fi"],
            dtype={"RID": str},
        )
        self.parcellation_file = (
            self.parcellation_file.query("Dataset == @dataset")
            .drop(columns=["Dataset", "PROGRESSION_CATEGORY"])
            .copy()
        )  # query the parcellations
        # (
        #     self.rids,
        #     self.time_obs,
        #     self.hit,
        #     self.age,
        #     self.mmse,
        #     self.sex,
        # ) = read_csv_demog(self.csvname)
        (
            self.rids,
            self.time_obs,
            self.hit,
            self.age,
            self.mmse,
            self.sex,
            self.educ,
            self.apoe,
        ) = read_csv_demog_ed_apoe(self.csvname)
        if dataset == "ADNI":
            data_order = glob.glob("/data1/RGAN_Data/Z/" + "*nii*")
        else:
            data_order = glob.glob("/data1/RGAN_Data/Z_E/" + "*nii*")

        rid_order = filter_rid(data_order)

        self.parcellation_file["RID"] = self.parcellation_file["RID"].apply(
            lambda x: x.zfill(4)
        )

        self.parcellation_file.set_index("RID", inplace=True)

        idx_map = [self.rids.index(x) for x in rid_order]

        # idx_map = [self.rids.index(x) for x in self.parcellation_file.index]

        self.parcellation_file = self.parcellation_file.loc[
            rid_order, :
        ].reset_index(
            drop=True
        )  # sort by RID
        self.rids = np.asarray(self.rids)[idx_map]
        self.time_obs = np.asarray(self.time_obs)[idx_map]
        self.hit = np.asarray(self.hit)[idx_map]

        if add_age:
            self.parcellation_file["age"] = np.asarray(self.age)[idx_map]
        if add_mmse:
            self.parcellation_file["mmse"] = np.asarray(self.mmse)[idx_map]
        if add_sex:
            self.parcellation_file["sex"] = np.asarray(self.sex)[idx_map]
        if add_apoe:
            self.parcellation_file["apoe"] = np.asarray(self.apoe)[idx_map]
        if add_educ:
            self.parcellation_file["educ"] = np.asarray(self.educ)[idx_map]

        self._cutoff(36.0)

        l = len(self.rids)
        split1 = int(l * ratio[0])
        split2 = int(l * (ratio[0] + ratio[1]))
        idxs = list(range(l))
        random.shuffle(idxs)
        if "train" in stage:
            self.index_list = idxs[:split1]
        elif "valid" in stage:
            self.index_list = idxs[split1:split2]
        elif "test" in stage:
            self.index_list = idxs[split2:]
        elif "all" in stage:
            self.index_list = idxs
            # print(len(self.index_list))
        else:
            raise Exception("Unexpected Stage for MLP_Data!")

        self._prep_data(self.parcellation_file)

    def _cutoff(self, n_months: float):
        valid_datapoints = [
            t > n_months or y == 1 for t, y in zip(self.time_obs, self.hit)
        ]

        self.rids = self.rids[valid_datapoints]
        self.hit = self.hit[valid_datapoints]
        self.time_obs = self.time_obs[valid_datapoints]
        self.parcellation_file = self.parcellation_file.loc[
            valid_datapoints, :
        ]
        self.PMCI = np.array(
            [t <= n_months and y == 1 for t, y in zip(self.time_obs, self.hit)]
        )
        self.PMCI = np.where(self.PMCI, 1, 0)

    def _prep_data(self, feature_df):
        self.rids = np.array(self.rids)
        feature_df.drop(
            columns=["CSF", "3thVen", "4thVen", "InfLatVen", "LatVen"],
            inplace=True,
        )  # drop ventricles
        self.labels = feature_df.columns
        self.data = feature_df.to_numpy()

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):
        idx_transformed = self.index_list[idx]
        x = self.data[idx_transformed]
        pmci = self.PMCI[idx_transformed]
        rids = self.rids[idx_transformed]
        return x, pmci, rids

    def get_features(self):
        return self.labels

    def get_data(self):
        return self.data

    def get_labels(self):
        return self.PMCI

if __name__ == "__main__":
    Data_dir_NACC = "/data2/MRI_PET_DATA/processed_images_final_cox_test/brain_stripped_cox_test/"
    Data_dir_ADNI = "/data2/MRI_PET_DATA/processed_images_final_cox_noqc/brain_stripped_cox_noqc/"
    train_data = B_Data(Data_dir_ADNI, stage='all', step_size=60)
    # self.train_data = B_Data(data_dir, stage="train", seed=self.seed, step_size=self.config["step_size"])
    # external_data = B_Data(Data_dir_NACC, stage='all', step_size=60, external=True)
    data = train_data
    # data = external_data
    # for _ in data:
    #     pass
    print(len(data))
    # external_data = ParcellationDataBinary(1, stage='all', dataset='ADNI', ratio=(0.6, 0.2, 0.2), add_age=False,
    #              add_mmse=False, partitioner=retrieve_kfold_partition)
    # print(len(external_data))
    # external_data = ParcellationDataBinary(1, stage='all', dataset='NACC', ratio=(0.6, 0.2, 0.2), add_age=False,
    #              add_mmse=False, partitioner=retrieve_kfold_partition)
    # print(len(external_data))

def filter_rid(file_list: list) -> list:
    rid_extract = lambda x: x.split("masked_brain_mri_")[1].split(".nii")[0]
    return list(map(rid_extract, file_list))
