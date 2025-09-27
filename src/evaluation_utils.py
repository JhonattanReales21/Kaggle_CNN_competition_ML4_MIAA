import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
from PIL import Image

from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import random

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF

import albumentations as A
from albumentations.pytorch import ToTensorV2


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

import os
import sys

sys.path.append("..")

from src.data_utils import *
from src.cnn_models import *
from src.training_utils import *


## ---------- Plotting history Utilities ---------- ##


def plot_history(history, figsize=(18, 5)):
    ep = [h["epoch"] for h in history]
    tr_loss = [h["train"]["loss"] for h in history]
    vl_loss = [h["valid"]["loss"] for h in history]
    tr_iou = [h["train"]["mean_iou"] for h in history]
    vl_iou = [h["valid"]["mean_iou"] for h in history]
    tr_acc = [h["train"]["acc"] for h in history]
    vl_acc = [h["valid"]["acc"] for h in history]

    # Crear figura con 3 subplots en fila
    fig, axs = plt.subplots(1, 3, figsize=figsize)

    # Loss
    axs[0].plot(ep, tr_loss, label="train")
    axs[0].plot(ep, vl_loss, label="valid")
    axs[0].set_title("Loss")
    axs[0].legend()

    # Accuracy
    axs[1].plot(ep, tr_acc, label="train")
    axs[1].plot(ep, vl_acc, label="valid")
    axs[1].set_title("Accuracy")
    axs[1].legend()

    # IoU
    axs[2].plot(ep, tr_iou, label="train")
    axs[2].plot(ep, vl_iou, label="valid")
    axs[2].set_title("IoU")
    axs[2].legend()

    plt.tight_layout()
    plt.show()


## ---------- Models Sanity check Utilities ---------- ##


def sanity_check_batch(trainer) -> None:
    """
    Perform a quick validation sanity check on the model using a random batch.

    This function:
      1. Randomly selects a batch from the validation dataloader.
      2. Runs the model in evaluation mode (no gradient computation).
      3. Prints statistics (min, max, mean) for target and predicted bounding boxes.
      4. Computes and reports the mean Intersection over Union (IoU) between
         predicted and ground truth boxes (scaled to pixel coordinates).

    Args:
        trainer: The training manager object that holds:
            - dl_valid: Validation dataloader.
            - model: Model to be evaluated.
            - cfg.img_size: Image size used for scaling boxes.

    Returns:
        None
    """

    # Convert validation dataloader into a list of batches (so we can sample randomly)
    dl_list = list(trainer.dl_valid)
    batch = random.choice(dl_list)  # Pick a random batch

    imgs, t_boxes01, t_cls, fn = (
        batch  # images, target boxes [0,1], target labels, filenames
    )

    # Move tensors to the correct device (GPU/CPU)
    imgs = imgs.to(DEVICE)
    t_boxes01 = t_boxes01.to(DEVICE)

    # Forward pass without gradient tracking (evaluation mode)
    with torch.no_grad():
        logits, p_boxes01 = trainer.model(
            imgs
        )  # Predictions in [0,1] normalized coords

    def stats(tensor: torch.Tensor) -> tuple[float, float, float]:
        """Return min, max, mean of a tensor as floats."""
        return float(tensor.min()), float(tensor.max()), float(tensor.mean())

    # Print statistics for targets and predictions
    print("Targets [0,1] min/max/mean:", stats(t_boxes01))
    print("Preds   [0,1] min/max/mean:", stats(p_boxes01))

    # Scale normalized boxes ([0,1]) to pixel coordinates
    img_size = float(trainer.cfg.img_size)
    pred_boxes = p_boxes01 * img_size
    gt_boxes = t_boxes01 * img_size

    # Compute IoU for each sample in the batch
    ious = [iou(pred_boxes[i], gt_boxes[i]).item() for i in range(len(pred_boxes))]
    print("IoU mean (batch):", float(sum(ious) / len(ious)))


def denorm(
    img_t: torch.Tensor,
    pers_norm: bool = False,
    mean_ls: list[float] | None = None,
    std_ls: list[float] | None = None,
) -> Image.Image:
    """
    Reverse the normalization of a tensor image and convert it to a PIL Image.

    Args:
        img_t (torch.Tensor): Normalized image tensor (C,H,W).
        pers_norm (bool): If True, use custom mean/std lists.
                          If False, default to ImageNet normalization values.
        mean_ls (list[float], optional): Custom mean values per channel (RGB).
        std_ls (list[float], optional): Custom std values per channel (RGB).

    Returns:
        Image.Image: The de-normalized image as a PIL Image.
    """

    if pers_norm:
        if mean_ls is None or std_ls is None:
            raise ValueError("Custom normalization requires both mean_ls and std_ls.")
        mean = torch.tensor(mean_ls)[:, None, None]
        std = torch.tensor(std_ls)[:, None, None]
    else:
        # Default ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]

    # Reverse normalization
    img = img_t.cpu() * std + mean

    # Clamp to valid range [0,1], convert to 0–255 uint8, then to HWC numpy
    img = (img.clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()

    return Image.fromarray(img)
