import torch
import torchmetrics
import torchmetrics.classification

class MeanPixelAccuracy(torchmetrics.Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("conf_matrix", default=torch.zeros(2, 2), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        if preds.shape != target.shape:
            raise ValueError("preds and target must have the same shape")

        confmat = torchmetrics.classification.BinaryConfusionMatrix()(preds, target)
        self.conf_matrix += confmat

    def compute(self) -> torch.Tensor:
        TN, FP, FN, TP = self.conf_matrix.flatten()
        return 1 / 2 * ( (TP / (TP + FN)) + (TN / (TN + FP)) )
