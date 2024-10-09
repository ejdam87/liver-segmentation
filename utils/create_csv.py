# Program for creating a CSV pairing inputs to the expected outputs
import pandas as pd


def make_csv(num_images: int, fold_in: str, fold_out: str, path_out: str) -> None:
    """
    <num_images> : number of images to save
    <fold_in>    : folder in which input images are stored
    <fold_out>   : folder in which output images are stored
    <path_out>   : where to store the output csv
    """
    # we assume a specific naming of images (output masks as well) used for our data
    X = [ f"{fold_in}/{str(i).zfill(3)}.png" for i in range(num_images) ]
    Y = [ f"{fold_out}/{str(i).zfill(3)}_mask.png" for i in range(num_images) ]

    df = pd.DataFrame( {"X": X, "Y": Y} )
    df.to_csv(path_out, index=False)


if __name__ == "__main__":
    make_csv(400, "data/Images", "data/Labels", "./data_pairs.csv")
