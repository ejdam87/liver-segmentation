# This file should contain the testing (evaluating) of the model
import torch
import torch.utils
import torch.utils.data
import torchmetrics
from tqdm import tqdm


def test_model(model: torch.nn.Module,
               test_data: torch.utils.data.DataLoader,
               test_metrics: dict[str, torchmetrics.Metric]) -> None:
    """
    Main function for model testing

    attributes:
    - model        : model to be tested
    - test_metrics : mapping from name of the metrics to the object computing the metric

    returns:
    Nothing, just prints the metric values.
    """

    print("Starting testing...")
    with torch.no_grad():
        metric_values = {name: 0 for name in test_metrics}
        n_batches = len(test_data)
        test_progress_bar = tqdm(test_data, desc="Test progress")

        for X_batch, Y_batch in test_progress_bar:
            pred = model(X_batch)

            for name, metric in test_metrics.items():
                metric_values[name] += metric( pred, Y_batch ).item()

        for name, val in metric_values.items():
            print( f"{name} = {val / n_batches : .3f}" )
