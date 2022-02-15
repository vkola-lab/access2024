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

def read_json(config_file):
    with open(config_file) as config_buffer:
        config = json.loads(config_buffer.read())
    return config

class B_Data(Dataset):
    #Brain data
    def __init__(self, data_dir, stage, ratio=(0.6, 0.2, 0.2), seed=1000, step_size=10, external=False, Pre=False):
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
        print('dataloader', data.shape)
        print('dataloader', g_data.shape)

        sys.exit()
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
    def __init__(self, data_dir, stage, ratio=(0.6, 0.2, 0.2), seed=1000, step_size=10, external=False):
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

def _read_csv_cox(filename, skip_ids: list=None):
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        fileIDs, time_obs, hit, age, mmse = [], [], [], [], []
        for r in reader:
            if skip_ids is not None:
                if r['RID'] in skip_ids:
                    continue
            fileIDs += [str(r['RID'])]
            time_obs += [float(r['TIMES'])]  # changed to TIMES_ROUNDED. consider switching so observations for progressors are all < 1 year
            hit += [int(float(r['PROGRESSES']))]
            age += [float(r['AGE'])]
            if 'MMSCORE_mmse' in r.keys():
                mmse += [float(r['MMSCORE_mmse'])]
            else:
                mmse += [np.nan if r['MMSE'] == '' else float(r['MMSE'])]
    return fileIDs, np.asarray(time_obs), np.asarray(hit), age, mmse

def _read_csv_csf(filename):
    parcellation_tbl = pd.read_csv(filename)
    valid_columns = ["abeta", "tau", "ptau"]
    parcellation_tbl = parcellation_tbl[valid_columns + ['RID']].copy()
    return parcellation_tbl

def _retrieve_kfold_partition(idxs, stage, folds=5, exp_idx=1, shuffle=True,
                              random_state=120):
    idxs = np.asarray(idxs).copy()
    if shuffle:
        np.random.seed(random_state)
        idxs = np.random.permutation(idxs)
    if 'all' in stage:
        return idxs
    if len(idxs.shape) > 1: raise ValueError
    fold_len = len(idxs) // folds
    folds_stitched = []
    for f in range(folds):
        folds_stitched.append(idxs[f*fold_len:(f+1)*fold_len])
    test_idx = exp_idx
    valid_idx = (exp_idx+1) % folds
    train_idx = np.setdiff1d(np.arange(0,folds,1),[test_idx, valid_idx])
    if 'test' in stage:
        return folds_stitched[test_idx]
    elif 'valid' in stage:
        return folds_stitched[valid_idx]
    elif 'train' in stage:
        return np.concatenate([folds_stitched[x] for x in train_idx], axis=0)
    else:
        raise ValueError

def deabbreviate_parcellation_columns(df):
    df_dict = pd.read_csv(
            './metadata/data_raw/neuromorphometrics/neuromorphometrics.csv',
                          usecols=['ROIabbr','ROIname'],sep=';')
    df_dict = df_dict.loc[[x[0] == 'l' for x in df_dict['ROIabbr']],:]
    df_dict['ROIabbr'] = df_dict['ROIabbr'].apply(
            lambda x: x[1:]
    )
    df_dict['ROIname'] = df_dict['ROIname'].apply(
            lambda x: x.replace('Left ', '')
    )
    df_dict = df_dict.set_index('ROIabbr').to_dict()['ROIname']
    df.rename(columns=df_dict, inplace=True)

def drop_ventricles(df, ventricle_list):
    df.drop(columns=ventricle_list, inplace=True)

def add_ventricle_info(parcellation_df, ventricle_df, ventricles):
    return parcellation_df.merge(ventricle_df[['RID'] + ventricles], on='RID',
                          validate="one_to_one")

class ParcellationDataMeta(Dataset):
    def __init__(self, seed, **kwargs):
        random.seed(1000)
        self.exp_idx = seed
        json_props = read_json('./simple_mlps/mlp_config'
                             '.json')
        self._json_props = json_props
        self.csv_directory = json_props['datadir']
        self.ventricles = json_props['ventricles']
        self.csvname = self.csv_directory + json_props['metadata_fi']
        self.parcellation_file = self.csv_directory + json_props['parcellation_fi']
        self.parcellation_file_csf = self.csv_directory + json_props[
            'parcellation_csf_fi']
        self.rids, self.time_obs, self.hit, self.age, self.mmse = \
            _read_csv_cox(self.csvname)

    def _prep_data(self, feature_df, stage):
        idxs = list(range(len(self.rids)))
        self.index_list = _retrieve_kfold_partition(idxs, stage, 5, self.exp_idx)
        self.rid = np.array(self.rids)
        logging.warning(f'selecting indices\n{self.rid[self.index_list]}\n\t '
                        f'for stage'
                        f'{stage} and random seed 1000')
        self.labels = feature_df.columns
        self.data_l = feature_df.to_numpy()
        self.data_l = torch.FloatTensor(self.data_l)
        self.data = self.data_l

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):
        idx_transformed = self.index_list[idx]
        x = self.data[idx_transformed]
        obs = self.time_obs[idx_transformed]
        hit = self.hit[idx_transformed]
        rid = self.rid[idx_transformed]
        return x, obs, hit, rid

    def get_features(self):
        return self.labels

    def get_data(self):
        return self.data

class ParcellationDataCSF(ParcellationDataMeta):
    def __init__(self, seed, stage, ratio=(0.6, 0.2, 0.2), add_age=False,
                 add_mmse=False):
        super().__init__(seed, stage=stage, ratio=ratio)
        parcellation_df = _read_csv_csf(self.csvname)
        parcellation_df['RID'] = parcellation_df['RID'].apply(
                lambda x: str(int(x)).zfill(4)
        )
        parcellation_df.set_index('RID', inplace=True)
        parcellation_df = parcellation_df.loc[self.rids,:].reset_index(
                drop=True)
        if add_age:
            parcellation_df['age'] = self.age
        if add_mmse:
            parcellation_df['mmse'] = self.mmse
        self._prep_data(parcellation_df, stage)

class ParcellationData(ParcellationDataMeta):
    def __init__(self, seed, stage, ratio=(0.6, 0.2, 0.2), add_age=False,
                 add_mmse=False):
        super().__init__(seed, stage=stage, ratio=ratio)
        parcellation_df = pd.read_csv(self.parcellation_file)
        parcellation_df['RID'] = parcellation_df['RID'].apply(
                lambda x: str(int(x)).zfill(4)
        )
        parcellation_df.set_index('RID', inplace=True)
        parcellation_df = parcellation_df.loc[self.rids,:].reset_index(
                drop=True)
        deabbreviate_parcellation_columns(parcellation_df)
        if add_age:
            parcellation_df['age'] = self.age
        if add_mmse:
            parcellation_df['mmse'] = self.mmse
        self._prep_data(parcellation_df, stage)

class ParcellationDataVentricles(ParcellationDataMeta):
    def __init__(self, seed, stage, ratio=(0.6, 0.2, 0.2),
                 ventricle_info=False, add_age=False,
                 add_mmse=False):
        super().__init__(seed, stage=stage)
        parcellation_df = pd.read_csv(self.parcellation_file, dtype={'RID':
                                                                         str})
        ventricle_df = pd.read_csv(self.parcellation_file_csf, dtype={'RID':
                                                                          str})
        drop_ventricles(parcellation_df, self.ventricles)
        if ventricle_info:
           parcellation_df = add_ventricle_info(parcellation_df, ventricle_df,
                                  self.ventricles)
        parcellation_df.set_index('RID', inplace=True)
        parcellation_df = parcellation_df.loc[self.rids,:].reset_index(
                drop=True)
        deabbreviate_parcellation_columns(parcellation_df)
        if add_age:
            parcellation_df['age'] = self.age
        if add_mmse:
            parcellation_df['mmse'] = self.mmse
        self._prep_data(parcellation_df, stage)

class ParcellationDataNacc(ParcellationDataMeta):
    def __init__(self, seed, stage, ratio=(0.6, 0.2, 0.2), add_age=False,
                 add_mmse=False):
        self.seed = seed
        random.seed(1000)
        json_props = read_json('./simple_mlps/mlp_config'
                             '.json')
        self._json_props = json_props
        self.csv_directory = json_props['datadir']
        self.csvname = self.csv_directory + json_props['metadata_fi_nacc']
        self.rids, self.time_obs, self.hit, self.age, self.mmse = \
            _read_csv_cox(
                self.csvname)
        csvname2 = self.csv_directory + json_props['parcellation_fi_nacc']
        parcellation_df = pd.read_csv(csvname2)
        parcellation_df.set_index('RID', inplace=True)
        parcellation_df = parcellation_df.loc[self.rids,:].reset_index(
                drop=True)
        if add_age:
            parcellation_df['age'] = self.age
        if add_mmse:
            parcellation_df['mmse'] = self.mmse
        self.exp_idx = 1
        deabbreviate_parcellation_columns(parcellation_df)
        self._prep_data(parcellation_df, stage)

class ParcellationDataVentriclesNacc(ParcellationDataMeta):
    def __init__(self, seed, stage, ratio=(0.6, 0.2, 0.2),
                 ventricle_info=False, add_age=False,
                 add_mmse=False):
        self.seed = seed
        random.seed(1000)
        json_props = read_json('./simple_mlps/mlp_config'
                             '.json')
        self._json_props = json_props
        self.ventricles = json_props['ventricles']
        self.csv_directory = json_props['datadir']
        self.csvname = self.csv_directory + json_props['metadata_fi_nacc']
        self.rids, self.time_obs, self.hit, self.age, self.mmse = \
            _read_csv_cox(self.csvname)
        csvname2 = self.csv_directory + json_props['parcellation_fi_nacc']
        csvname3 = self.csv_directory + json_props['parcellation_csf_fi_nacc']
        parcellation_df = pd.read_csv(csvname2, dtype={'RID': str})
        ventricle_df = pd.read_csv(csvname3, dtype={'RID': str})
        drop_ventricles(parcellation_df, self.ventricles)
        if ventricle_info:
            parcellation_df = add_ventricle_info(parcellation_df, ventricle_df,
                                  self.ventricles)
        parcellation_df.set_index('RID', inplace=True)
        parcellation_df = parcellation_df.loc[self.rids,:].reset_index(
                drop=True)
        if add_age:
            parcellation_df['age'] = self.age
        if add_mmse:
            parcellation_df['mmse'] = self.mmse
        deabbreviate_parcellation_columns(parcellation_df)
        self.exp_idx = 1
        self._prep_data(parcellation_df, stage)

if __name__ == "__main__":
    Data_dir_NACC = "/data2/MRI_PET_DATA/processed_images_final_cox_test/brain_stripped_cox_test/"
    Data_dir_ADNI = "/data2/MRI_PET_DATA/processed_images_final_unused_cox/brain_stripped_unused_cox/"
    external_data = B_Data(Data_dir_ADNI, 'all', Pre=True)
    print(len(external_data))
    for _ in external_data:
        print()
        sys.exit()
