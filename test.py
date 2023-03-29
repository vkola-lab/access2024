import os
import unittest
from typing import Literal

import numpy as np

from dataloader import (
    ParcellationDataBinary,
    filter_rid,
    read_csv,
    read_csv_ext,
)


def _test_individual_fold(rids, fold, stage, ds_) -> None:
    txt = _read_txt(fold, stage, ds_)
    return np.array_equal(np.asarray(rids), np.asarray(txt))


def _read_txt(fold, stage, ds_) -> list:
    with open(
        f"./rids/{ds_}/{stage}{fold*100}.txt", "r", encoding="utf-8"
    ) as fi_:
        payload = fi_.readlines()
    file_list = (
        payload[0]
        .replace("[", "")
        .replace("]", "")
        .replace("\n", "")
        .replace("'", "")
        .split(", ")
    )
    return filter_rid(file_list)


def _process(str_) -> list:
    return list(
        map(lambda x: float(x.replace("\n", "").split("__")[-1]), str_)
    )


class TestDataLoader(unittest.TestCase):
    def test_label(self) -> None:
        path2 = "/home/mfromano/Research/rcgan/checkpoint_dir"
        path = "/home/xzhou/rcgan/checkpoint_dir/"
        payload1 = ""
        payload2 = ""
        for i in range(5):
            with open(
                f"{path}/CNN_Standard_CG_{10+i}/raw_score_test_{i}.txt",
                "r",
                encoding="utf-8",
            ) as fi:
                payload1 = _process(fi.readlines())
            with open(
                f"{path2}/mlp_bce_{i}_exp{i}/raw_score_ADNI_{i}_test.txt",
                "r",
                encoding="utf-8",
            ) as fi2:
                payload2 = _process(fi2.readlines())
            self.assertListEqual(payload1, payload2)

    def test_hits(self) -> None:
        for ds_ in ("ADNI", "NACC"):
            ds = ParcellationDataBinary(0, stage="all", dataset=ds_)
            csvname = "./csvs/merged_dataframe_cox_noqc_pruned_final.csv"
            csvname = os.path.expanduser(csvname)
            rids, pmci = read_csv(csvname)  # training file
            if ds_ == "NACC":
                csvname = "./csvs/merged_dataframe_cox_test_pruned_final.csv"
                rids, pmci = read_csv_ext(csvname)  # training file
            orig_ids, orig_time_hit = ds.rids, ds.PMCI
            orig_dict = {rid: hit for rid, hit in zip(orig_ids, orig_time_hit)}
            new_dict = {rid: hit for rid, hit in zip(rids, pmci)}
            self.assertTrue(
                all(
                    orig_dict[key] == new_dict[key] for key in orig_dict.keys()
                )
            )

    def test_folds(self) -> None:
        for fold in range(5):
            for stage in (
                "train",
                "test",
                "valid",
            ):
                for ds_ in ("ADNI",):
                    dl_ = ParcellationDataBinary(
                        fold, stage=stage, dataset=ds_, seed=fold * 100
                    )
                    rids = dl_.rids[dl_.index_list]
                    self.assertTrue(
                        _test_individual_fold(rids, fold, stage, ds_)
                    )


if __name__ == "__main__":
    unittest.main()
