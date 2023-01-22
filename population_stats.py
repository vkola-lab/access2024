""" 
Split up data into progress vs non-progress; get stats for the following:

1. sex
2. mmse
3. age
4. apoe #
5. educ (# years)
"""

from typing import Literal

import numpy as np
import pandas as pd

from dataloader import Dataset, ParcellationDataBinary

DatasetName = Literal["ADNI", "NACC"]


def load_dataset(dataset: DatasetName) -> Dataset:
    """
    Loads complete dataset using the same dataloader used to construct the MLP model.

    Args:
        dataset (str): NACC or ADNI

    Returns:
        Dataset: Dataset with age, mmse, sex all added
    """
    return ParcellationDataBinary(
        exp_idx=0,
        stage="all",
        dataset=dataset,
        ratio=(0.8, 0.1, 0.1),
        add_age=True,
        add_mmse=True,
        add_sex=True,
        add_apoe=True,
        add_educ=True,
    )


def load_dataset_as_df(dataset: DatasetName) -> pd.DataFrame:
    """
    Retrieve only the values for age, mmse, sex, and ?progress
    for each participant in the dataset "dataset"


    Args:
        dataset (DatasetName): either NACC or ADNI

    Returns:
        pd.DataFrame: df with age, mmse, sex, and ?progress
    """
    ds_ = load_dataset(dataset)
    fields_of_interest = ["age", "mmse", "sex", "progress"]
    labels = ds_.get_labels()
    data = ds_.get_data()
    data_of_interest = np.concatenate(
        [data[:, -3:], np.expand_dims(labels, 1)], axis=1
    )
    df_ = pd.DataFrame(data_of_interest, columns=fields_of_interest)
    return df_


def stack_datasets(**kwargs) -> pd.DataFrame:
    """
    From a list of datasetname: pd.DataFrame,
        concatenate the associated pd.DataFrames and create a
        column "Dataset" storing the dataframe label

    Returns:
        pd.DataFrame: concatenated dataframe
    """
    datasets = []
    for label, tbl in kwargs.items():
        tbl["Dataset"] = label
        datasets.append(tbl)
    return pd.concat(datasets, axis=0, ignore_index=True)


def demo_stats(df: pd.DataFrame) -> None:
    """
    Retrieve demographic statistics, both mean / std for each and cross tabulated data for sex

    Args:
        df (pd.DataFrame): output from stack_datasets
    """
    with open("demo_stats.txt", "w") as fi:
        fi.write("MEAN\n\n")
        fi.write(
            str(df.groupby(["Dataset", "progress"]).agg(lambda x: np.mean(x)))
        )
        fi.write("\n\nSTD\n\n")
        fi.write(
            str(df.groupby(["Dataset", "progress"]).agg(lambda x: np.std(x)))
        )
        fi.write("\n\nN\n\n")
        fi.write(
            str(
                df.groupby(["Dataset", "progress"]).agg(
                    lambda x: np.sum([~np.isnan(y) for y in x])
                )
            )
        )
        fi.write("Sex")
        adni_ = df.query("Dataset == 'adni'").copy()
        nacc_ = df.query("Dataset == 'nacc'").copy()
        fi.write("\n\nADNI\n")
        fi.write(str(pd.crosstab(adni_["sex"], adni_["progress"])))
        fi.write("\n\nNACC\n")
        fi.write(str(pd.crosstab(nacc_["sex"], nacc_["progress"])))


if __name__ == "__main__":
    nacc = load_dataset_as_df("NACC")
    adni = load_dataset_as_df("ADNI")
    ds = stack_datasets(nacc=nacc, adni=adni)
    demo_stats(ds)
