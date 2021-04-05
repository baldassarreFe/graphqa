import numpy as np
import torch.utils
import wandb
from matplotlib import pyplot as plt


def figure_to_image(fig: plt.Figure, close: bool):
    img = torch.utils.tensorboard._utils.figure_to_image(fig, close=close)
    img = np.moveaxis(img, 0, -1)
    img = wandb.Image(img)
    return img
