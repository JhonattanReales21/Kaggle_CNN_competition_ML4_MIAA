import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
from PIL import Image
import os

CLASS_TO_LABEL = {"mask": 1, "no-mask": 0}
LABEL_TO_CLASS = {1: "mask", 0: "no-mask"}

DEFAULT_COLS = {
    "filename": "filename",
    "xmin": "xmin",
    "ymin": "ymin",
    "xmax": "xmax",
    "ymax": "ymax",
    "class": "class",
}


def compute_mean_std_image_dir(image_dir, image_filenames):
    """
    Compute the mean and std of a list of raw images in a directory.
    """
    transform = transforms.ToTensor()
    means, stds = [], []

    for fname in image_filenames:
        # Open image and convert to RGB (3 channels)
        img = Image.open(os.path.join(image_dir, fname)).convert("RGB")

        # Convert to tensor: shape (3, H, W)
        tensor = transform(img)

        # Compute mean and std for each channel (R, G, B) across H and W
        means.append(tensor.mean(dim=[1, 2]))
        stds.append(tensor.std(dim=[1, 2]))

    # Aggregate results across the whole dataset
    mean = torch.stack(means).mean(0)  # average per channel
    std = torch.stack(stds).mean(0)  # std per channel

    # Return as Python lists
    return mean.tolist(), std.tolist()


def show_raw_images_with_bboxes(df, indices, img_dir, cols=3):
    n = len(indices)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = axes.flatten() if n > 1 else [axes]

    for ax in axes:
        ax.axis("off")  # hide axes by default

    for i, idx in enumerate(indices):
        row = df.iloc[idx]
        filename = row["filename"]
        xmin, ymin, xmax, ymax = row["xmin"], row["ymin"], row["xmax"], row["ymax"]
        label = row["class"]

        # open image
        img_path = os.path.join(img_dir, filename)
        image = Image.open(img_path).convert("RGB")

        # Show image
        axes[i].imshow(image)

        # Draw bounding box
        rect = patches.Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            linewidth=2,
            edgecolor="red",
            facecolor="none",
        )
        axes[i].add_patch(rect)

        # Draw label
        axes[i].text(xmin, ymin - 5, label, color="red", fontsize=12, weight="bold")

        axes[i].set_title(f"Index {idx}", fontsize=12)

    plt.tight_layout()
    plt.show()


def show_dataset_sample(dataset, idx, class_map=None, figure_size=(6, 6)):
    """
    Visualize a sample from a pytorch Dataset object with its bounding box and label.

    Args:
        dataset: instance of a pytorch Dataset object.
        idx (int): index of the sample to visualize.
        class_map (dict, optional): maps class ids -> class names.
    """
    # ---- Get sample from dataset ----
    img_t, box, label, fname = dataset[idx]

    # ---- Denormalize image ----
    # (C,H,W) tensor back to numpy (H,W,C) in [0,1] range
    mean = torch.tensor(dataset.mean)[:, None, None]
    std = torch.tensor(dataset.std)[:, None, None]
    img_denorm = img_t * std + mean
    img_np = img_denorm.permute(1, 2, 0).cpu().numpy()
    img_np = img_np.clip(0, 1)  # ensure valid range for display

    # ---- Convert normalized bbox [0,1] to pixel coordinates ----
    h, w = img_np.shape[:2]
    x1, y1, x2, y2 = box.numpy()
    x1, y1, x2, y2 = x1 * w, y1 * h, x2 * w, y2 * h

    # ---- Get class name ----
    try:
        cls_id = int(label.item())
    except:
        cls_id = label[0]
    cls_name = class_map[cls_id] if class_map else str(cls_id)

    # ---- Plot image and bounding box ----
    fig, ax = plt.subplots(1, figsize=figure_size)
    ax.imshow(img_np)

    # Draw bbox rectangle
    rect = patches.Rectangle(
        (x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="green", facecolor="none"
    )
    ax.add_patch(rect)

    # Add class label text above the box
    ax.text(
        x1,
        max(0, y1 - 5),
        cls_name,
        color="green",
        fontsize=12,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
    )

    # Set title and remove axes
    ax.set_title(f"Index {idx} - {cls_name} - {fname[:10]}", fontsize=12)
    ax.axis("off")
    plt.show()


## ---- Dataset classes ----
class MaskDataset(Dataset):
    def __init__(
        self,
        df,
        images_dir,
        img_size=256,
        cols=DEFAULT_COLS,
        personalized_norm=False,
        custom_mean=None,
        custom_std=None,
    ):
        """
        Args:
            df (pd.DataFrame): dataframe with annotations (filename, xmin, ymin, xmax, ymax, class).
            images_dir (Path): path to the folder containing images.
            img_size (int): resize dimension for the images (output will be square img_size x img_size).
            cols (dict): dictionary with column names mapping.
            personalized_norm (bool):
                - False -> use ImageNet mean/std.
                - True  -> use dataset-specific mean/std (must provide custom_mean and custom_std).
            custom_mean, custom_std (list): lists with 3 values each (R, G, B).
        """
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.img_size = img_size
        self.cols = cols

        # Normalization values
        if personalized_norm:
            if custom_mean is None or custom_std is None:
                raise ValueError(
                    "custom_mean and custom_std must be provided if personalized_norm=True"
                )
            self.mean = custom_mean
            self.std = custom_std
        else:
            # Standard ImageNet normalization values
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]

        # Transformations: only Resize -> ToTensor -> Normalize
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )

    def _load_image(self, filename: str) -> Image.Image:
        """Open an image from disk and convert to RGB."""
        path = os.path.join(self.images_dir, filename)
        return Image.open(path).convert("RGB")

    def __len__(self):
        """Return the total number of samples."""
        return len(self.df)

    def __getitem__(self, idx):
        """
        Load a single sample:
        - Read image
        - Resize and normalize
        - Rescale bounding box to the new image size
        - Return (image_tensor, target_box, target_class, filename)
        """
        row = self.df.iloc[idx]
        fname = row[self.cols["filename"]]

        # Original bounding box values
        x1 = float(row[self.cols["xmin"]])
        y1 = float(row[self.cols["ymin"]])
        x2 = float(row[self.cols["xmax"]])
        y2 = float(row[self.cols["ymax"]])
        cls_name = str(row[self.cols["class"]]).strip().lower()
        label = CLASS_TO_LABEL.get(cls_name, 0)

        # Load image and apply transformations
        img = self._load_image(fname)
        w0, h0 = img.size
        img_t = self.transform(img)

        # Rescale bounding box to match resized image
        sx = self.img_size / float(w0)
        sy = self.img_size / float(h0)
        bx = [x1 * sx, y1 * sy, x2 * sx, y2 * sy]
        bx = np.clip(bx, 0, self.img_size - 1)  # keep values inside image bounds

        # Normalize bounding box coordinates to [0,1] - relative positions
        s = float(self.img_size)
        target_box = torch.tensor(
            [bx[0] / s, bx[1] / s, bx[2] / s, bx[3] / s], dtype=torch.float32
        )

        # Class label (float tensor)
        target_cls = torch.tensor([float(label)], dtype=torch.float32)

        return img_t, target_box, target_cls, fname


class AugmentedDataset(torch.utils.data.Dataset):
    """
    A dataset wrapper that applies Albumentations augmentations
    to an existing dataset while handling normalization properly.

    Args:
        base_dataset: The original dataset (must return (image_tensor, bbox, label, filename)).
        aug: Albumentations augmentation pipeline (can be None for no augmentation).
        mean: Normalization mean values (ImageNet defaults).
        std: Normalization std values (ImageNet defaults).
    """

    def __init__(
        self,
        base_dataset,
        aug=None,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ):
        self.base_dataset = base_dataset
        self.aug = aug
        self.mean = mean
        self.std = std

    def __len__(self):
        # Length is the same as the base dataset
        return len(self.base_dataset)

    def __getitem__(self, idx):
        # Get item from the base dataset
        img_t, box, label, fname = self.base_dataset[idx]

        # If no augmentation pipeline is provided, return directly
        if self.aug is None:
            return img_t, box, label, fname

        # ---- Denormalize tensor to numpy [0–255] ----
        # Albumentations expects images as uint8 NumPy arrays in [0–255] range
        # And (H, W, C) format
        img = img_t.clone()
        mean = torch.tensor(self.mean)[:, None, None]
        std = torch.tensor(self.std)[:, None, None]
        img = img * std + mean
        img = (img.clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()

        # ---- Rescale bbox to pixel values ----
        s = self.base_dataset.img_size
        box_px = [box[0] * s, box[1] * s, box[2] * s, box[3] * s]

        # ---- Albumentations transform ----
        transformed = self.aug(
            image=img,
            bboxes=[box_px],  # Albumentations expects bboxes as lists
            labels=[label.item()],  # Labels must be in list format
        )
        img_np = transformed["image"]
        bboxes = transformed["bboxes"]
        labels = transformed["labels"]

        # ---- Back to tensor (normalized again) ----
        img_t = transforms.ToTensor()(img_np)  # convert NumPy (H,W,C) → tensor (C,H,W)
        img_t = transforms.Normalize(self.mean, self.std)(
            img_t
        )  # re-apply normalization

        # ---- Handle bbox and label ----
        if len(bboxes):
            # Normalize bbox back to [0,1]
            bx = bboxes[0]
            box_out = torch.tensor(
                [bx[0] / s, bx[1] / s, bx[2] / s, bx[3] / s],
                dtype=torch.float32,
            )

            label_out = torch.tensor(labels, dtype=torch.float32)
        else:
            # If bbox or labels were removed by augmentation, fallback to original
            box_out = box
            label_out = label
            print(f"⚠️ Empty bboxes or labels for index {idx} after augmentation.")

        return img_t, box_out, label_out, fname
