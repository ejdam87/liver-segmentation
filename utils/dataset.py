import torch
from numpy.typing import NDArray

from utils.im_handle import load_image

# TODO -> try image patching and class balancing in the later stage

class ImageDataset(torch.utils.data.Dataset):

    def __init__(self, X: NDArray, Y: NDArray) -> None:
        super().__init__()
        assert len(X) == len(Y), "The number of inputs and expected outputs (labels) must be equal"
        self.X = X
        self.Y = Y

    def __len__(self) -> int:
        """
        Returns number of samples
        """
        return len(self.X)

    def __getitem__(self, i: int) -> tuple[ torch.Tensor, torch.Tensor ]:
        """
        Returns i-th sample as a pair of input and expected output
        """
        return load_image( self.X[i] ), load_image( self.Y[i] )
