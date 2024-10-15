# This file should handle the training of the model
import torch
import torch.utils
import torch.utils.data
import torchmetrics
from tqdm import tqdm


def train_model(model: torch.nn.Module,
                optimizer: torch.optim.Optimizer,
                train_data: torch.utils.data.DataLoader,
                val_data: torch.utils.data.DataLoader,
                epochs: int,
                loss: torch.nn.Module,
                val_metrics: dict[ str, torchmetrics.Metric ]) -> None:
    """
    Main function for training models

    attributes:
    - model       : model to be trained
    - optimizer   : optimizer to be used
    - train_data  : data to train on
    - val_data    : data to validate on
    - epochs      : number of times to go over train_data
    - loss        : loss function to use
    - val_metrics : mapping names of metrics to actual object performing the measurements during validation

    returns:
    The function does not return anything, just modifies model's parameters.
    """

    for epoch in range(epochs):

        progress_bar = tqdm(train_data, desc=f"Epoch {epoch+1}/{epochs}")
        print(f"Starting epoch n.{epoch} of training...")
        for X_batch, Y_batch in progress_bar:
            optimizer.zero_grad()
            output = model(X_batch)
            l = loss(output, Y_batch)
            l.backward()
            optimizer.step()

        print("Starting validation...")
        with torch.no_grad():   # no need of gradient for validation
            metric_values = {name: 0 for name in val_metrics}
            n_batches = len(val_data)

            for X_batch, Y_batch in val_data:
                pred = model(X_batch)

                for name, metric in val_metrics.items():
                    metric_values[name] += metric( pred, Y_batch )

            for name, val in metric_values.items():
                print( f"{name} = {val / n_batches : .3f}" )

        print()
