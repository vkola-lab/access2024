from dataloader import ParcellationDataBinary, retrieve_kfold_partition, Dataset
from tabulate import tabulate
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

'''
Split up data into progress vs non-progress
need:

1. sex
2. mmse
3. age

'''

def load_dataset(Dataset: str) -> Dataset:
    return ParcellationDataBinary(
        exp_idx=0, 
        stage='all',
        dataset=Dataset, 
        ratio=(0.6, 0.2, 0.2), 
        add_age=True,
        add_mmse=True,
        add_sex=True,
        partitioner=retrieve_kfold_partition)

def load_dataset_as_df(Dataset: str) -> pd.DataFrame:
    ds = load_dataset(Dataset)
    fields_of_interest = ['age','mmse','sex', 'progress']
    labels = ds.get_labels()
    data = ds.get_data()
    data_of_interest = np.concatenate([data[:,-3:], np.expand_dims(labels,1)], axis=1)
    df = pd.DataFrame(data_of_interest, columns=fields_of_interest)
    return df

def stack_datasets(**kwargs) -> pd.DataFrame:
    datasets = []
    for label, tbl in kwargs.items():
        tbl['Dataset'] = label
        datasets.append(tbl)
    return pd.concat(datasets, axis=0, ignore_index=True)

def demo_stats(df: pd.DataFrame):
    with open('demo_stats.txt', 'w') as fi:
        fi.write('MEAN\n\n')
        fi.write(str(df.groupby(['Dataset','progress']).agg(lambda x: np.mean(x))))
        fi.write('\n\nSTD\n\n')
        fi.write(str(df.groupby(['Dataset','progress']).agg(lambda x: np.std(x))))
        fi.write('\n\nN\n\n')
        fi.write(str(df.groupby(['Dataset','progress']).agg(lambda x: np.sum([~np.isnan(y) for y in x]))))
        fi.write('Sex')
        adni = df.query('Dataset == \'adni\'').copy()
        nacc = df.query('Dataset == \'nacc\'').copy()
        fi.write('\n\nADNI\n')
        fi.write(str(pd.crosstab(adni['sex'], adni['progress'])))
        fi.write('\n\nNACC\n')
        fi.write(str(pd.crosstab(nacc['sex'], nacc['progress'])))


if __name__ == '__main__':
    nacc = load_dataset_as_df('NACC')
    adni = load_dataset_as_df('ADNI')
    ds = stack_datasets(nacc=nacc, adni=adni)
    demo_stats(ds)