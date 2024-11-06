# This file should provide functionality to perform single image segmentation
import torch
from utils.im_handle import load_image, save_image

def single_im_pred(model: torch.nn.Module, t: float, in_path: str, out_path: str, device: str) -> None:
    """
    Function to save a prediction of <model> on image stored in <in_path>, then binarize
    it using threshold <t> and save it to <out_path>.
    """
    t_im = load_image(in_path)
    bt_im = t_im.unsqueeze(0)
    bt_im = bt_im.to(device)

    bpred_mask = None
    with torch.no_grad():
        bpred_mask = model(bt_im)

    pred_mask = bpred_mask.squeeze(0)
    bin_mask = pred_mask > t
    bin_mask = bin_mask.to(dtype=torch.uint8) * 255
    bin_mask = bin_mask.to("cpu")
    save_image(bin_mask, out_path)
