import torchvision.transforms as T
import torch.nn.functional as F
from torchvision.transforms.functional import rgb_to_grayscale
import torch.nn as nn
import torch
from modules.loss import L1CharbonnierLoss

# %%


def img2dft(x):
    x_gray = rgb_to_grayscale(x)
    x_fft = torch.fft.fft2(x_gray, dim=(-2, -1))
    x_shift = torch.fft.fftshift(x_fft)
    dft = torch.log(torch.abs(x_shift) + torch.tensor(1, dtype=x.dtype).to(x.device))
    return dft


class DFTDistortionLoss(nn.Module):
    """Custom distortion loss with a Lagrangian parameter."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        if self.cfg.l1_charbonnier_loss:
            self.diff_loss = L1CharbonnierLoss()
        else:
            self.diff_loss = nn.MSELoss()
        self.lmbda = cfg.lmbda
        self.lmbda_dft = cfg.loss.lmbda_dft

    def forward(self, output, target):
        output_dft = img2dft(output["x_hat"])
        target_dft = img2dft(target)
        output["mse_loss"] = self.diff_loss(output["x_hat"], target)
        # output["dft_loss"] = torch.mean(torch.abs(output_dft - target_dft)) # mse loss?
        output["dft_loss"] = self.diff_loss(output_dft, target_dft)
        dft_loss = output["dft_loss"] * self.lmbda_dft * (255**2)

        output["loss"] = (255**2) * output["mse_loss"] * self.lmbda + dft_loss
        return output
