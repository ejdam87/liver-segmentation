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
                val_metrics: dict[ str, torchmetrics.Metric ],
                device: str) -> None:
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
    - Dictionary containing loss and metric values for each epoch
    """

    train_n_batches = len(train_data)
    val_n_batches = len(val_data)

    train_loss = []
    val_loss = []
    metric_vals_epochs = { name : [] for name in val_metrics }

    for epoch in range(epochs):

        print(f"Starting epoch n.{epoch + 1} of training...")
        epoch_loss = 0
        progress_bar = tqdm(train_data, desc=f"Epoch {epoch + 1}/{epochs}")

        for X_batch, Y_batch in progress_bar:
            X_batch_c, Y_batch_c = X_batch.to(device), Y_batch.to(device)

            optimizer.zero_grad()
            output = model(X_batch_c)
            l = loss(output, Y_batch_c)
            l.backward()
            optimizer.step()

            epoch_loss += l.item()

        train_loss.append(epoch_loss / train_n_batches)

        print("Starting validation...")
        epoch_loss = 0
        with torch.no_grad():   # no need of gradient for validation
            metric_values = {name: 0 for name in val_metrics}

            val_progress_bar = tqdm(val_data, desc="Validation progress")
            for X_batch, Y_batch in val_progress_bar:
                X_batch_c, Y_batch_c = X_batch.to(device), Y_batch.to(device)

                pred = model(X_batch_c)

                epoch_loss += loss(pred, Y_batch_c).item()

                for name, metric in val_metrics.items():
                    metric_values[name] += metric( pred, Y_batch_c ).item()

            val_loss.append( epoch_loss / val_n_batches )

            for name, val in metric_values.items():
                print( f"{name} = {val / val_n_batches : .3f}" )
                metric_vals_epochs[name] = val / val_n_batches

        print()
    
    metric_vals_epochs["train_loss"] = train_loss
    metric_vals_epochs["val_loss"] = val_loss
    return metric_vals_epochs
