""" 
Split up data into progress vs non-progress; get stats for the following:

1. sex
2. mmse
3. age
4. apoe #
5. educ (# years)
"""

import itertools
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

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
    fields_of_interest = ["age", "mmse", "sex", "apoe", "educ", "progress"]
    labels = ds_.get_labels()
    data = ds_.get_data()
    data_of_interest = np.concatenate(
        [data[:, -5:], np.expand_dims(labels, 1)], axis=1
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

        fi.write("\n\nAPOE")
        fi.write("\n\nADNI\n")
        fi.write(str(pd.crosstab(adni_["apoe"], adni_["progress"])))
        fi.write("\n\nNACC\n")
        fi.write(str(pd.crosstab(nacc_["apoe"], nacc_["progress"])))


def demo_stats_quant():
    raise NotImplementedError


class ChiSquare:
    def __init__(self, cross_tab):
        self.df = cross_tab
        self.columns = np.setdiff1d(list(cross_tab.columns), ["All"])
        self.rows = np.setdiff1d(list(cross_tab.index), ["All"])
        self.columns_name = cross_tab.columns.name
        assert len(self.rows) == 2
        if len(self.columns) == 2:
            self.omnibus_st = fishertest(
                cross_tab.loc[self.rows, self.columns]
            )
            self.omnibus_str = (
                f"Fisher test for {self.columns_name}"
                + _stringify_fisher(self.omnibus_st)
            )
        else:
            self.omnibus_st = chisq(cross_tab.loc[self.rows, self.columns])
            self.omnibus_str = (
                f"Omnibus for {self.columns_name}"
                + _stringify_chisq(self.omnibus_st)
            )
        proportions = self.df.loc[self.rows, self.columns].divide(
            self.df.loc[self.rows, "All"], axis="rows"
        )
        self.proportions = pd.melt(
            proportions.reset_index(),
            id_vars="Dataset",
            value_name="Proportion",
        ).set_index(["Dataset", self.columns_name])
        self.chisq_pairwise()

    def chisq_pairwise(self):
        self.pairwise_stats = {}
        self.pairwise_str = ""
        all_values = self.df.loc[self.rows, "All"].to_numpy().reshape((-1, 1))
        if len(self.columns) < 3:
            return
        for col in self.columns:
            current_col = (
                self.df.loc[self.rows, col].to_numpy().reshape((-1, 1))
            )
            current_tbl = np.concatenate(
                [current_col, all_values - current_col], axis=1
            )
            self.pairwise_stats[col] = fishertest(
                current_tbl, nreps=len(self.columns)
            )
            self.pairwise_str += (
                f"\n{self.columns_name},col {str(col)}: \n"
                + _stringify_fisher(self.pairwise_stats[col])
            )
            self.pairwise_str += (
                "\tproportion ADNI vs NACC: {} vs {}\n".format(
                    self.proportions.loc["adni", col].values[0],
                    self.proportions.loc["nacc", col].values[0],
                )
            )
            self.pairwise_str += (
                "\tcounts for ADNI vs NACC: {} vs {}\n".format(
                    self.df.loc["adni", "All"], self.df.loc["nacc", "All"]
                )
            )

    def __str__(self):
        return self.omnibus_str + "\n" + self.pairwise_str


def pairwise_mannwhitneyu(df, col, group_col):
    factors = pd.unique(df[group_col])
    n_pairs = len(list(itertools.combinations(factors, 2)))
    factor_pairs = itertools.combinations(factors, 2)
    mwu_output_strings = []
    for pair in factor_pairs:
        pair_1, pair_2 = pair
        x = df.loc[df[group_col] == pair_1, col].copy().to_numpy()
        y = df.loc[df[group_col] == pair_2, col].copy().to_numpy()
        mwu = MannWhitneyU(x, y, pair_1, pair_2, n_comparisons=n_pairs)
        mwu_output_strings.append(str(mwu))
    return mwu_output_strings


def chisq(tbl, nreps=1):
    chi2, p, dof, expected = stats.chi2_contingency(tbl)
    if any(expected.reshape((-1, 1)) < 5):
        chi2, p, dof, expected = stats.chi2_contingency(tbl, correction=True)
    return {
        "chi2": chi2,
        "p": p * nreps,
        "dof": dof,
        "expected_lt5": any(expected.reshape((-1, 1)) < 5),
    }


def _stringify_chisq(chisq_dict):
    return "".join(
        [f"\t{str(key)}={value}\n" for key, value in chisq_dict.items()]
    )


def fishertest(tbl, nreps=1):
    st, p = stats.fisher_exact(tbl)

    return {
        "st": st,
        "p": p * nreps,
    }


def _stringify_fisher(fisher_dict):
    return "".join(
        [f"\t{str(key)}={value}\n" for key, value in fisher_dict.items()]
    )


class MannWhitneyU:
    def __init__(self, x, y, x_label, y_label, n_comparisons=1):
        self.x, self.y = x.reshape((-1, 1)), y.reshape((-1, 1))
        self.x_label, self.y_label = x_label, y_label
        self.n_comparisons = n_comparisons
        self.mannwhitneyu()

    def mannwhitneyu(self):
        x, y = self.x, self.y
        x_nan = sum(np.isnan(x))[0]
        n_x = sum(~np.isnan(x))[0]
        y_nan = sum(np.isnan(y))[0]
        n_y = sum(~np.isnan(y))[0]
        _, p_less = stats.mannwhitneyu(
            x[~np.isnan(x)], y[~np.isnan(y)], alternative="less"
        )
        stat, p_greater = stats.mannwhitneyu(
            x[~np.isnan(x)], y[~np.isnan(y)], alternative="greater"
        )
        if p_less * 2 * self.n_comparisons < 0.05:
            output_str = "<"
        elif p_greater * 2 * self.n_comparisons < 0.05:
            output_str = ">"
        else:
            output_str = "="
        output_str = f"{self.x_label}{output_str}{self.y_label}"
        self.stats = {
            "stat": stat,
            "p": min([min([p_less, p_greater]) * 2 * self.n_comparisons, 1]),
            "n_x": n_x,
            "n_y": n_y,
            "n_x_nan": x_nan,
            "n_y_nan": y_nan,
        }
        self.string = output_str

    def __str__(self):
        str_list = [f"Wilcoxon Rank-Sum test: {self.string}\n"] + [
            f"\t{key}={value}\n" for key, value in self.stats.items()
        ]
        return "".join(str_list)


def demo_stats_ttest(df: pd.DataFrame) -> None:
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

        fi.write("\n\nAPOE")
        fi.write("\n\nADNI\n")
        fi.write(str(pd.crosstab(adni_["apoe"], adni_["progress"])))
        fi.write("\n\nNACC\n")
        fi.write(str(pd.crosstab(nacc_["apoe"], nacc_["progress"])))


if __name__ == "__main__":
    nacc = load_dataset_as_df("NACC")
    adni = load_dataset_as_df("ADNI")
    ds = stack_datasets(nacc=nacc, adni=adni)
    demo_stats(ds)
