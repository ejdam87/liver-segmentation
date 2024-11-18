import torch
import torchmetrics
import torchmetrics.classification

class MeanIoU(torchmetrics.Metric):
    def __init__(self, device, **kwargs):
        super().__init__(**kwargs)
        self.add_state("conf_matrix", default=torch.zeros(2, 2), dist_reduce_fx="sum")
        self.bcm = torchmetrics.classification.BinaryConfusionMatrix().to(device=device)

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        if preds.shape != target.shape:
            raise ValueError("preds and target must have the same shape")

        confmat = self.bcm(preds, target)
        self.conf_matrix += confmat

    def compute(self) -> torch.Tensor:
        TN, FP, FN, TP = self.conf_matrix.flatten()
        return 1 / 2 * ( (TP / (TP + FP + FN)) + (TN / (TN + FP + FN)) )
