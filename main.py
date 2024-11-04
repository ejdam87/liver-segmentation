# main file (only this file should be run in the final version) serving as an interface for the user
import sys
import json
import torch
import torchmetrics
import pandas as pd
import numpy as np
import torchmetrics.classification
from typing import Any
from torch.utils.data import DataLoader

from models.unet import UNet
from utils.dataset import ImageDataset
from utils.persistency import save_model, load_model
from train import train_model
from test import test_model
from single_im import single_im_pred


TRAIN_CONF = "./configs/train_config.json"
TEST_CONF = "./configs/test_config.json"
PRED_CONF = "./configs/pred_config.json"

MODEL_DICT = {
    "unet": UNet
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
    "accuracy": torchmetrics.classification.BinaryAccuracy,
    "recall": torchmetrics.classification.BinaryRecall
}


def load_conf(path: str) -> dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def main_train():

    # use GPU computing power if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    conf = load_conf(TRAIN_CONF)

    model = MODEL_DICT[conf["model"]]( conf["in_dim"], conf["out_dim"],  ACTIVATION_DICT[ conf["out_activation"] ]() )
    model = model.to(device)

    optim = OPTIM_DICT[conf["optim"]](model.parameters(), lr=conf["lr"])
    loss = LOSS_DICT[conf["loss"]]()
    X = pd.read_csv( conf["data_x_path"] ).to_numpy()
    Y = pd.read_csv( conf["data_y_path"] ).to_numpy()

    # --- Shuffling and splitting training data into train and validation sets
    data = np.column_stack( (X, Y) )
    np.random.shuffle(data)
    split_index = int( conf["validation_size"] * len(data) )

    val_x = data[:split_index, 0]
    val_y = data[:split_index, 1]
    train_x = data[split_index:, 0]
    train_y = data[split_index:, 1]
    # ---

    train_dataset = ImageDataset(train_x, train_y)
    val_dataset = ImageDataset(val_x, val_y)

    train_dataloader = DataLoader(train_dataset, batch_size=conf["train_batch_size"], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=conf["valid_batch_size"])

    val_metrics = { m : METRIC_DICT[m]().to(device) for m in conf["val_metrics"] }
    train_model(model, optim, train_dataloader, val_dataloader, conf["epochs"], loss, val_metrics, device)

    if conf["save_model"]:
        save_model(model, "trained_model.pth")


def main_test():
    conf = load_conf(TRAIN_CONF)
    model = MODEL_DICT[conf["model"]]( conf["in_dim"], conf["out_dim"],  ACTIVATION_DICT[ conf["out_activation"] ]() )
    load_model( model, conf["model_path"] )

    X = pd.read_csv( conf["data_x_path"] ).to_numpy()
    Y = pd.read_csv( conf["data_y_path"] ).to_numpy()
    test_dataset = ImageDataset(X, Y)
    loader = DataLoader(test_dataset, batch_size=conf["batch_size"])
    test_metrics = { m : METRIC_DICT[m] for m in conf["test_metrics"] }

    test_model(model, loader, test_metrics)


def main_pred():
    conf = load_conf(PRED_CONF)
    model = MODEL_DICT[conf["model"]]( conf["in_dim"], conf["out_dim"],  ACTIVATION_DICT[ conf["out_activation"] ]() )
    load_model( model, conf["model_path"] )
    single_im_pred(model, 0.5, conf["image_in_path"], "pred.png")


if __name__ == "__main__":

    if len(sys.argv) == 1:
        print("provide one of the following arguments: train/test/pred")
    elif sys.argv[1] == "train":
        main_train()
    elif sys.argv[1] == "test":
        main_test()
    elif sys.argv[1] == "pred":
        main_pred()
    else:
        print("Option not recognized")
