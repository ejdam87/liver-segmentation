import torch
import torchvision.transforms as transforms
from PIL import Image

import numpy as np

def load_image(path: str) -> torch.tensor:
    """
    Returns image located at <path> as tensor
    """
    im = Image.open(path).convert("L")
    transform = transforms.ToTensor()
    return transform(im)


def save_image(im: torch.tensor, path: str) -> None:
    """
    Stores tensor image <im> to <path>
    """
    to_pil = transforms.ToPILImage()
    out = to_pil(im)
    out.save(path)
