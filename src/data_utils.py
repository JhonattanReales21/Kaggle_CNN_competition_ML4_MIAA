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
    Compute the mean and std of a list of images in a directory.
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


def show_images_with_bboxes(df, indices, img_dir, cols=3):
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
    def __init__(self, base_dataset, train_compose, train=True):
        """
        Wrapper around MaskDataset to apply Albumentations.

        Args:
            base_dataset: an instance of MaskDataset (returns img_t, bbox, cls, fname).
            train (bool): if True, apply augmentations; if False, only normalization.
            train_compose: albumentations.Compose object with desired augmentations.
        """
        self.base_dataset = base_dataset
        self.train = train
        self.img_size = base_dataset.img_size  # needed for bbox scaling

        if self.train:
            self.augment = train_compose
        else:
            self.augment = A.Compose(
                [],
                bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
            )

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        # Get original sample from MaskDataset
        img_t, target_box, target_cls, fname = self.base_dataset[idx]

        # Convert tensor (C, H, W) -> numpy (H, W, C) in [0,255]
        img_np = (img_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        # Convert normalized box [0,1] -> absolute Pascal VOC [xmin, ymin, xmax, ymax]
        x_min = target_box[0].item() * self.img_size
        y_min = target_box[1].item() * self.img_size
        x_max = target_box[2].item() * self.img_size
        y_max = target_box[3].item() * self.img_size
        box_abs = [x_min, y_min, x_max, y_max]

        # Apply augmentations
        augmented = self.augment(
            image=img_np, bboxes=[box_abs], labels=[int(target_cls.item())]
        )

        aug_img = augmented["image"]
        aug_boxes = augmented["bboxes"]
        aug_labels = augmented["labels"]

        # Convert back to torch tensor
        aug_img = transforms.ToTensor()(aug_img)

        if len(aug_boxes) == 0:
            # No box survived augmentations -> fallback
            aug_box = target_box
            aug_label = target_cls
        else:
            # Convert absolute box back to normalized [0,1]
            bx = np.array(aug_boxes[0], dtype=np.float32)
            aug_box = torch.tensor(
                [
                    bx[0] / self.img_size,
                    bx[1] / self.img_size,
                    bx[2] / self.img_size,
                    bx[3] / self.img_size,
                ],
                dtype=torch.float32,
            )
            aug_label = torch.tensor([float(aug_labels[0])], dtype=torch.float32)

        return aug_img, aug_box, aug_label, fname


# from torch.utils.data import DataLoader

# # Create your base dataset
# train_dataset = MaskDataset(df, Path("data/images"), img_size=256,
#                             personalized_norm=False, cols=DEFAULT_COLS)

# # Wrap with augmentations
# train_dataset_aug = AugmentedDataset(train_dataset, train_compose, train=True)

# # Validation dataset (no augmentations)
# val_dataset = AugmentedDataset(train_dataset, train=False)

# # Dataloaders
# train_loader = DataLoader(train_dataset_aug, batch_size=16, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
