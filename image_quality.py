import json
import csv
import os, sys
import matlab.engine
import collections

import numpy as np

from scipy.interpolate import interp1d
from dataloader import B_IQ_Data
from utils import iqa_tensor, p_val, SSIM
from torch.utils.data import DataLoader
from tabulate import tabulate

class Image_Quality:
    def __init__(self):
        self.step_size = 2
        self.eng = matlab.engine.start_matlab()
        # in_size = 121*145*121

    def prepare_dataloader(self, data_dir, ext):
        # maybe 'all' for all datasets????????
        stage = 'test'
        if ext:
            stage = 'all'
        data  = B_IQ_Data(data_dir, stage=stage, seed=1000, step_size=self.step_size, external=ext)
        # return DataLoader(data, batch_size=1, shuffle=False)
        return data

    def iqa(self, metrics=['CNR', 'SNR', 'brisque', 'niqe', 'piqe', 'ssim'], names=['T', 'Z', 'G', 'CG_1', 'CG_2'], datasets=['', '_E']):
        data_root = '/data1/RGAN_Data/'
        iqa_dict = collections.defaultdict(list)
        for ds in datasets:
            data_dir = data_root
            dataset = self.prepare_dataloader(data_dir, ds)
            for datas in dataset:
                for m in metrics:
                    for n, d in zip(names, datas):
                        if m == 'ssim':
                            #first is the target!
                            # print(SSIM(datas[0], datas[1]))
                            # print(SSIM(d, d))
                            # print('ssim')
                            # sys.exit()
                            iqa_dict[n+'_'+m+'_'+ds].append(SSIM(datas[0], d))
                        else:
                            iqa_dict[n+'_'+m+'_'+ds].append(iqa_tensor(d, self.eng, m))
                    # p_va = p_val(list1, list2)
        
        table = [metrics]
        for ds in datasets:
            for n in names:
                line = [n+ds]
                for m in metrics:
                    line.append('{0:.3f}+/-{1:.3f}'.format(np.mean(iqa_dict[n+'_'+m+'_'+ds]), np.std(iqa_dict[n+'_'+m+'_'+ds])))
                table.append(line)
        print(tabulate(table, headers='firstrow'))
        
        json_w = json.dumps(iqa_dict)
        f = open('iqa_dict.json', 'w')
        f.write(json_w)
        f.close()
    
    def pr(self):
        with open('iqa_dict.json') as json_file:
            iqa_dict = json.load(json_file)
        
        metrics=['CNR', 'SNR', 'brisque', 'piqe', 'ssim']
        names=['T', 'Z', 'G', 'CG_1', 'CG_2']
        datasets=['', '_E']
        table = [metrics]
        for ds in datasets:
            for n in names:
                line = [n+ds]
                for m in metrics:
                    line.append('{0:.3f}+/-{1:.3f}'.format(np.mean(iqa_dict[n+'_'+m+'_'+ds]), np.std(iqa_dict[n+'_'+m+'_'+ds])))
                table.append(line)
        print(tabulate(table, headers='firstrow'))
        

if __name__ == "__main__":
    print('Image Quality Analysis:')
    iq = Image_Quality()
    # iq.iqa()
    iq.pr()