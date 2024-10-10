import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def tt_split(seed: int, tsize: int, in_path: str, out_path: str) -> None:
    """
    Function to split the data into train and test sets (<tsize> gives a fraction of a test set).
    For reproducibility purposes, we set a fixed <seed> and save the
    result of split in <out_path> as CSV file.
    """
    
    df = pd.read_csv(in_path)
    X = df["X"]
    Y = df["Y"]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=seed, test_size=tsize)
    X_train.to_csv(f"{out_path}/train_x.csv", index=False)
    X_test.to_csv(f"{out_path}/test_x.csv", index=False)
    y_train.to_csv(f"{out_path}/train_y.csv", index=False)
    y_test.to_csv(f"{out_path}/test_y.csv", index=False)

if __name__ == "__main__":
    tt_split(seed=42, tsize=0.1, in_path="data/data_pairs.csv", out_path="data/")
