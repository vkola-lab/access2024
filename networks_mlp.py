from dataloader import ParcellationDataBinary
from models import _MLP_Surv
from utils import write_raw_score
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict
import tabulate
import os
import matplotlib.pyplot as plt
from torchviz import make_dot
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd

torch.set_num_threads(
    1
)  # may lower training speed (and potentially overheat issue) if machine has many cpus!
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")


class MLP_Wrapper:
    def __init__(
        self,
        exp_idx,
        model_name,
        lr,
        weight_decay,
        model,
        model_kwargs,
        add_age=False,
        add_mmse=False,
    ):
        self._age = add_age
        self._mmse = add_mmse
        self.seed = exp_idx * 100
        self.exp_idx = exp_idx
        self.model_name = model_name
        self.device = device
        self.dataset = ParcellationDataBinary
        self.lr = lr
        self.weight_decay = weight_decay
        self.c_index = []
        self.checkpoint_dir = "./checkpoint_dir/{}_exp{}/".format(
            self.model_name, self.exp_idx
        )
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.prepare_dataloader()
        torch.manual_seed(self.seed)
        self.criterion = nn.BCELoss().to(self.device)
        self.model = model(in_size=self.in_size, **model_kwargs).float()
        self.model.to(self.device)

    def prepare_dataloader(self):
        kwargs = dict(
            exp_idx=self.exp_idx, seed=self.seed, add_age=self._age, add_mmse=self._mmse
        )
        self.train_data = self.dataset(stage="train", dataset="ADNI", **kwargs)
        self.features = self.train_data.get_features()
        self.in_size = len(self.features)
        self.valid_data = self.dataset(stage="valid", dataset="ADNI", **kwargs)
        self.test_data = self.dataset(stage="test", dataset="ADNI", **kwargs)
        self.all_data = self.dataset(stage="all", dataset="ADNI", **kwargs)
        self.nacc_data = self.dataset(stage="all", dataset="NACC", **kwargs)
        self.train_dataloader = DataLoader(
            self.train_data, batch_size=len(self.train_data)
        )
        self.valid_dataloader = DataLoader(
            self.valid_data, batch_size=len(self.valid_data)
        )
        self.test_dataloader = DataLoader(
            self.test_data, batch_size=len(self.test_data)
        )
        self.all_dataloader = DataLoader(
            self.all_data, batch_size=len(self.all_data), shuffle=False
        )
        self.nacc_dataloader = DataLoader(
            self.nacc_data, batch_size=len(self.nacc_data)
        )

    def load(self):
        state = None
        for _, _, files in os.walk(self.checkpoint_dir):
            for file in files:
                if file.endswith(".pth"):
                    try:
                        state = torch.load(self.checkpoint_dir + file)
                    except Exception as exc:
                        raise FileNotFoundError(self.checkpoint_dir + file) from exc
        if state is not None:
            self.model.load_state_dict(state)
        else:
            raise FileNotFoundError(self.checkpoint_dir)

    def save_checkpoint(self, loss):
        score = loss
        if score <= self.optimal_valid_metric:
            self.optimal_epoch = self.epoch
            self.optimal_valid_metric = score
            for root, _, Files in os.walk(self.checkpoint_dir):
                for File in Files:
                    if File.endswith(".pth"):
                        try:
                            os.remove(self.checkpoint_dir + File)
                        except:
                            pass
            torch.save(
                self.model.state_dict(),
                "{}{}_{}.pth".format(
                    self.checkpoint_dir, self.model_name, self.optimal_epoch
                ),
            )

    def train(self, epochs):
        self.train_loss = []
        self.optimal_valid_matrix = None
        self.optimal_valid_metric = np.inf
        self.optimal_epoch = -1
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            betas=(0.5, 0.999),
            weight_decay=self.weight_decay,
        )
        writer = SummaryWriter()
        for self.epoch in range(epochs):
            self.train_model_epoch(self.optimizer)
            val_loss = self.valid_model_epoch()
            self.save_checkpoint(val_loss)
            writer.add_scalar("Loss/validate", val_loss.item(), self.epoch)
        self.optimal_path = "{}{}_{}.pth".format(
            self.checkpoint_dir, self.model_name, self.optimal_epoch
        )
        print("Location: {}".format(self.optimal_path))
        print(self.optimal_valid_metric)
        return self.optimal_valid_metric

    def train_model_epoch(self, optimizer):
        self.model.train()
        for data, pmci, _ in self.train_dataloader:
            self.model.zero_grad()
            preds = self.model(data.to(self.device).float())
            loss = self.criterion(
                preds.squeeze(), pmci.to(self.device).float().squeeze()
            )
            if self.epoch == 0:
                graph = make_dot(
                    loss,
                    params=dict(self.model.named_parameters()),
                    show_attrs=True,
                    show_saved=True,
                )
                graph.render("mlp_graph")
                plt.close()

            loss.backward()
            optimizer.step()

    def valid_model_epoch(self):
        with torch.no_grad():
            self.model.eval()
            for data, pmci, _ in self.valid_dataloader:
                preds = self.model(data.to(self.device).float())
                loss = self.criterion(
                    preds.squeeze(), pmci.to(self.device).float().squeeze()
                )
        return loss

    def retrieve_testing_data(self, external_data, fold="all"):
        if external_data:
            dataloader = self.nacc_dataloader
        else:
            if fold == "all":
                dataloader = self.all_dataloader
            elif fold == "valid":
                dataloader = self.valid_dataloader
            elif fold == "test":
                dataloader = self.test_dataloader
            elif fold == "train":
                dataloader = self.train_dataloader
            else:
                raise ValueError(f"Invalid fold specified: {fold}")
        with torch.no_grad():
            self.load()
            self.model.eval()
            for data, pmci, rids in dataloader:
                preds = self.model(data.to(self.device).float()).to("cpu")
                rids = rids
            return preds, pmci, rids

    def test_surv_data_optimal_epoch(self, external_data=False, fold="all"):
        key = "ADNI" if not external_data else "NACC"
        preds, pmci, rids = self.retrieve_testing_data(external_data, fold)
        preds_raw = preds.squeeze()
        preds = torch.round(preds).squeeze()
        report = classification_report(
            y_true=pmci,
            y_pred=preds,
            target_names=[key + fold + "_0", key + fold + "_1"],
            labels=[0, 1],
            zero_division=1,
            output_dict=True,
        )
        f = open(
            self.checkpoint_dir + f"raw_score_{key}_{self.exp_idx}_{fold}.txt",
            "w",
        )
        write_raw_score(f, preds_raw, pmci)
        with open(f"rids/mlp_{key}_{fold}_{self.seed}.txt", "w") as fi:
            for rid, pred in zip(rids, preds):
                fi.write(str(rid) + "," + str(pred) + "\n")
        return report


def run(load: bool = True) -> Dict[str, list]:
    mlp_list = []
    mlp_output = {"NACCall": [], "ADNItrain": [], "ADNIvalid": [], "ADNItest": []}
    for exp in range(5):
        mlp = MLP_Wrapper(
            exp_idx=exp,
            model_name=f"mlp_bce_{exp}",
            lr=0.01,
            weight_decay=0,
            model=_MLP_Surv,
            model_kwargs=dict(drop_rate=0.5, fil_num=100, output_shape=1),
        )
        mlp_list.append(mlp)
    for mlp in mlp_list:
        if load:
            mlp.load()
        else:
            mlp.train(1000)
            mlp.load()
        mlp_output["NACCall"].append(
            mlp.test_surv_data_optimal_epoch(external_data=True)
        )
        mlp_output["ADNItrain"].append(
            mlp.test_surv_data_optimal_epoch(external_data=False, fold="train")
        )
        mlp_output["ADNIvalid"].append(
            mlp.test_surv_data_optimal_epoch(external_data=False, fold="valid")
        )
        mlp_output["ADNItest"].append(
            mlp.test_surv_data_optimal_epoch(external_data=False, fold="test")
        )
    return mlp_output


def tabulate_report(report: dict, dataset: str) -> Dict[str, list]:
    report = report[dataset]
    label1 = f"{dataset}_1"
    label0 = f"{dataset}_0"
    value_list = {
        label0: [],
        label1: [],
        "accuracy": [],
        "weighted avg": [],
        "macro avg": [],
    }
    for label in value_list:
        for idx in report:
            value_list[label].append(pd.Series(idx[label]))
        value_list[label] = pd.concat(value_list[label], axis=1).T.describe()
        value_list[label] = value_list[label].loc[["mean", "std"], :].copy()
    with open(f"checkpoint_dir/results_bce_{dataset}.txt", "w") as fi:
        for label in value_list:
            fi.write("\n\n" + label + "\n")
            fi.write(
                tabulate.tabulate(
                    value_list[label],
                    headers=value_list[label].columns,
                    showindex="always",
                )
            )
    return value_list


def run_and_tabulate(load) -> None:
    g = run(load)
    for fold in ["train", "valid", "test"]:
        tabulate_report(g, "ADNI" + fold)
    tabulate_report(g, "NACCall")


if __name__ == "__main__":
    run_and_tabulate(False)
