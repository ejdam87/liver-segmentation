import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def tt_split(seed: int, tsize: int, x_path: str, y_path: str, out_path: str) -> None:
    """
    Function to split the data into train and test sets (<tsize> gives a fraction of a test set).
    For reproducibility purposes, we set a fixed <seed> and save the
    result of split in <out_path> as CSV file.
    """

    dfx = pd.read_csv(x_path)
    dfy = pd.read_csv(y_path)
    X = dfx["X"]
    Y = dfy["Y"]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=seed, test_size=tsize)
    X_train.to_csv(f"{out_path}/train_x.csv", index=False)
    X_test.to_csv(f"{out_path}/val_x.csv", index=False)
    y_train.to_csv(f"{out_path}/train_y.csv", index=False)
    y_test.to_csv(f"{out_path}/val_y.csv", index=False)

if __name__ == "__main__":
    tt_split(seed=42, tsize=0.2, x_path="data/train_x.csv", y_path="data/train_y.csv", out_path="data/")
