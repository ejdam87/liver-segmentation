# main file (only this file should be run in the final version) serving as an interface for the user
import sys
import os
import json
import torch
import pandas as pd
import numpy as np
from typing import Any
from torch.utils.data import DataLoader

import utils.mpa as mpa
import utils.miou as miou
from models.unet import UNet
from models.baseline import BaselineCNN
from utils.dataset import ImageDataset
from utils.persistency import save_model, load_model
from train import train_model
from test import test_model
from single_im import single_im_pred


TRAIN_CONF = "./configs/train_config.json"
TEST_CONF = "./configs/test_config.json"
PRED_CONF = "./configs/pred_config.json"

MODEL_DICT = {
    "unet": UNet,
    "baseline": BaselineCNN,
}

LOSS_DICT = {
    "bce": torch.nn.BCELoss
}

ACTIVATION_DICT = {
    "sigmoid": torch.nn.Sigmoid
}

OPTIM_DICT = {
    "adam": torch.optim.Adam
}

METRIC_DICT = {
    "mPA" : mpa.MeanPixelAccuracy(device="cuda" if torch.cuda.is_available() else "cpu"),
    "mIoU": miou.MeanIoU(device="cuda" if torch.cuda.is_available() else "cpu")
}


def load_conf(path: str) -> dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def main_train(device: str) -> None:
    conf = load_conf(TRAIN_CONF)

    np.random.seed(conf["seed"])

    model = MODEL_DICT[conf["model"]]( conf["in_dim"], conf["out_dim"],  ACTIVATION_DICT[ conf["out_activation"] ]() )
    model = model.to(device)

    optim = OPTIM_DICT[conf["optim"]](model.parameters(), lr=conf["lr"])
    loss = LOSS_DICT[conf["loss"]]()

    train_x = pd.read_csv( conf["tdata_x_path"] ).to_numpy()[:, 0]
    train_y = pd.read_csv( conf["tdata_y_path"] ).to_numpy()[:, 0]
    val_x = pd.read_csv( conf["vdata_x_path"] ).to_numpy()[:, 0]
    val_y = pd.read_csv( conf["vdata_y_path"] ).to_numpy()[:, 0]

    train_dataset = ImageDataset(train_x, train_y)
    val_dataset = ImageDataset(val_x, val_y)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=conf["train_batch_size"],
                                  shuffle=True,
                                  num_workers=conf["num_workers"])

    val_dataloader = DataLoader(val_dataset,
                                batch_size=conf["valid_batch_size"],
                                num_workers=conf["num_workers"])

    val_metrics = { m : METRIC_DICT[m].to(device) for m in conf["val_metrics"] }
    vals = train_model(model, optim, train_dataloader, val_dataloader, conf["epochs"], loss, val_metrics, device)

    if conf["save_model"]:
        os.makedirs("train_output", exist_ok=True)
        save_model(model, "./train_output/trained_model.pth")
        pd.DataFrame(vals).to_csv("./train_output/metric_values.csv")


def main_test(device: str) -> None:
    conf = load_conf(TEST_CONF)
    model = MODEL_DICT[conf["model"]]( conf["in_dim"], conf["out_dim"],  ACTIVATION_DICT[ conf["out_activation"] ]() )
    load_model( model, conf["model_path"] )
    model = model.to(device)

    X = pd.read_csv( conf["data_x_path"] ).to_numpy()
    Y = pd.read_csv( conf["data_y_path"] ).to_numpy()

    X = X[:, 0]
    Y = Y[:, 0]

    test_dataset = ImageDataset(X, Y)
    loader = DataLoader(test_dataset, batch_size=conf["batch_size"])
    test_metrics = { m : METRIC_DICT[m].to(device) for m in conf["test_metrics"] }

    vals = test_model(model, loader, test_metrics, device)
    if conf["save_metrics"]:
        os.makedirs("test_output", exist_ok=True)
        pd.DataFrame(vals).to_csv("./test_output/metric_values.csv")


def main_pred(device: str) -> None:
    conf = load_conf(PRED_CONF)
    model = MODEL_DICT[conf["model"]]( conf["in_dim"], conf["out_dim"],  ACTIVATION_DICT[ conf["out_activation"] ]() )
    load_model( model, conf["model_path"] )
    model = model.to(device)

    os.makedirs("pred_output", exist_ok=True)
    single_im_pred(model, conf["threshold"], conf["image_in_path"], conf["image_out_path"], device)


if __name__ == "__main__":

    # use GPU computing power if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if len(sys.argv) == 1:
        print("provide one of the following arguments: train/test/pred")
    elif sys.argv[1] == "train":
        main_train(device)
    elif sys.argv[1] == "test":
        main_test(device)
    elif sys.argv[1] == "pred":
        main_pred(device)
    else:
        print("Option not recognized")
