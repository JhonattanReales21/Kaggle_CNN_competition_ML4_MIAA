import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
from PIL import Image
import cv2
from typing import List, Tuple, Union
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import random
from types import SimpleNamespace

import plotly
import optuna
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
    plot_slice,
    plot_contour,
    plot_pareto_front,
)

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
    print("IoU mean (Random batch):", float(sum(ious) / len(ious)))


## ---------- Plotting images Utilities ---------- ##


def denorm(
    imgs_t: Union[torch.Tensor, List[torch.Tensor]],
    pers_norm: bool = False,
    mean_ls: list[float] | None = None,
    std_ls: list[float] | None = None,
) -> List[Image.Image]:
    """
    Reverse the normalization of one or more tensor images and convert them to PIL Images.

    Args:
        imgs_t (torch.Tensor or List[torch.Tensor]):
            - Single normalized image tensor (C,H,W), or
            - List of image tensors.
        pers_norm (bool): If True, use custom mean/std lists.
                          If False, default to ImageNet normalization values.
        mean_ls (list[float], optional): Custom mean values per channel (RGB).
        std_ls (list[float], optional): Custom std values per channel (RGB).

    Returns:
        List[Image.Image]: List of de-normalized images as PIL Images.
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

    def _denorm_single(img_t: torch.Tensor) -> Image.Image:
        """Denormalize a single tensor and convert to PIL Image."""
        img = img_t.cpu() * std + mean
        img = (img.clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
        return Image.fromarray(img)

    # Handle both single tensor and list of tensors
    if isinstance(imgs_t, torch.Tensor):
        return [_denorm_single(imgs_t)]
    elif isinstance(imgs_t, list):
        return [_denorm_single(img) for img in imgs_t]
    else:
        raise TypeError("imgs_t must be a torch.Tensor or a list of torch.Tensor.")


def draw_annotations(
    imgs: List[np.ndarray],
    bboxes: List[List[Tuple[int, int, int, int]]],
    classes: List[List[int]] = None,
    colors: Union[Tuple[int, int, int], List[Tuple[int, int, int]]] = (0, 255, 0),
    thickness: int = 2,
    font_scale: int = 1,
    origin: Union[str, Tuple[int, int]] = (10, 30),  # could be "closed" or (x, y)
    prefix: str = "",
    id2obj: dict = None,
) -> List[np.ndarray]:
    """
    Draw bounding boxes and optionally class labels on a list of images.

    Args:
        imgs (List[np.ndarray]):
            List of images in BGR format (OpenCV-style).
        bboxes (List[List[Tuple[int, int, int, int]]]):
            List of lists of bounding boxes per image.
            Each bounding box is (xmin, ymin, xmax, ymax) in pixels.
        classes (List[List[int]], optional):
            List of lists of class IDs per image. If None, only boxes are drawn.
        colors (Tuple[int,int,int] or List[Tuple[int,int,int]], optional):
            Single BGR color for all boxes/text, or a list of colors (one per box).
            Default is green (0,255,0).
        thickness (int, optional):
            Thickness of bounding box lines. Default is 2.
        font_scale (int, optional):
            Font size for class labels. Default is 1.
        origin (str or tuple, optional):
            - "closed": place text near the bounding box (xmin, ymin + 5).
            - (x, y): fixed coordinates.
            Default is (10, 30).
        prefix (str, optional):
            String prefix to prepend to each class name (e.g., "pred: "). Default is "".
        id2obj (dict, optional):
            Mapping from class ID to class name. Required if `classes` is provided.

    Returns:
        List[np.ndarray]: List of images with annotations drawn.

    Notes:
        - If `colors` is a single tuple, the same color is used for all annotations.
        - If `classes` is provided, its length per image must match the number of bboxes.
        - This function modifies the input images in place.
    """
    annotated_imgs = []

    for img_idx, img in enumerate(imgs):
        img_copy = img.copy()
        boxes = bboxes[img_idx]
        cls_ids = classes[img_idx] if classes is not None else [None] * len(boxes)

        # Expand single color to list if needed
        if isinstance(colors, tuple):
            color_list = [colors] * len(boxes)
        else:
            color_list = colors

        for (xmin, ymin, xmax, ymax), cls_id, color in zip(boxes, cls_ids, color_list):
            # Draw bounding box
            cv2.rectangle(img_copy, (xmin, ymin), (xmax, ymax), color, thickness)

            # Draw class label if provided
            if cls_id is not None and id2obj is not None:
                class_name = id2obj.get(cls_id, str(cls_id))

                # text position
                if isinstance(origin, str) and origin == "closed":
                    text_pos = (xmin, max(ymin + 10, 10))  # 10px above the box
                elif isinstance(origin, tuple):
                    text_pos = origin
                else:
                    raise ValueError("origin must be 'closed' or a tuple (x, y)")

                cv2.putText(
                    img_copy,
                    f"{prefix}{class_name}",
                    text_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    color,
                    1,
                    cv2.LINE_AA,
                )

        annotated_imgs.append(img_copy)

    return annotated_imgs


def show_image_grid(imgs: List[np.ndarray], n_cols: int = 2, figsize=(12, 8)) -> None:
    """
    Display a list of images in a grid.

    Args:
        imgs (List[np.ndarray]): List of images (BGR format).
        n_cols (int): Number of columns in the grid.
        figsize (tuple): Size of the matplotlib figure.

    Returns:
        None
    """
    if len(imgs) == 0:
        raise ValueError("No images provided.")

    n_rows = int(np.ceil(len(imgs) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.array(axes).reshape(-1)  # Flatten axes for easy iteration

    for i, ax in enumerate(axes):
        if i < len(imgs):
            # Convert BGR (OpenCV) to RGB (matplotlib)
            img_rgb = imgs[i][..., ::-1]
            ax.imshow(img_rgb)
            # ax.set_title(f"Image {i+1}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def visualize_predictions(
    model_class,
    checkpoint_path: str,
    norm_mean: List[float],
    norm_std: List[float],
    valid_dataset: Dataset,
    n_images: int = 3,
    n_cols: int = 3,
    figsize=(12, 8),
    annotations_dict: dict = None,
    threshold: float = 0.5,
    device: str = "cuda",
):
    """
    Load a model checkpoint, run predictions on a random validation sample,
    and visualize images with ground-truth and predicted bounding boxes.

    Args:
        model_class (nn.Module): The model class (inherits nn.Module).
        checkpoint_path (str): Path to the .pt checkpoint file.
        norm_mean (List[float]): Mean values for normalization (R, G, B).
        norm_std (List[float]): Std values for normalization (R, G, B).
        valid_dataset (Dataset): Validation dataset.
        n_images (int): Number of random images to visualize from the batch.
        n_cols (int): Number of columns in the visualization grid.
        figsize (tuple): Size of the matplotlib figure.
        annotations_dict (dict, optional): Additional annotations to display.
        threshold (float): Classification threshold (default=0.5).
        device (str): Device to run inference on ("cuda" or "cpu").

    Returns:
        None. Displays a grid of annotated images.
    """
    # ------------------ Load model and checkpoint ------------------
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = SimpleNamespace(**ckpt["cfg"])  # recover cfg as Namespace

    # Instantiate model and load state dict
    if isinstance(model_class, nn.Module):
        model = model_class.to(device)  # Is already an instance
    else:
        model = model_class().to(device)  # Is a class
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # ------------------ Pick random validation sample(s) ------------------
    if n_images is None or n_images == 1:
        # Choose a single random image
        idx = random.randrange(len(valid_dataset))
        imgs, t_boxes01, t_cls, fnames = valid_dataset[idx]

    else:
        # Choose a random subset of size n_images
        idxs = random.sample(range(len(valid_dataset)), n_images)

        imgs_list, t_boxes_list, t_cls_list, fnames_list = [], [], [], []
        for i in idxs:
            img, box, cls, fname = valid_dataset[i]
            imgs_list.append(img)
            t_boxes_list.append(box)
            t_cls_list.append(cls)
            fnames_list.append(fname)

        # Stack tensors (tensors must have same shape)
        imgs = torch.stack(imgs_list)
        t_boxes01 = torch.stack(t_boxes_list)
        t_cls = torch.stack(t_cls_list)
        fnames = fnames_list

    # Move to device
    imgs, t_boxes01, t_cls = imgs.to(device), t_boxes01.to(device), t_cls.to(device)

    print("# of images selected:", len(fnames))

    # ------------------ Forward pass ------------------
    with torch.no_grad():
        logits, p_boxes01 = model(imgs)
        probs = torch.sigmoid(logits)
        if probs.ndim > 1 and probs.shape[1] == 1:
            probs = probs.squeeze(1)
        pred_cls = (probs > threshold).long()

    # ------------------ Convert boxes to pixel coordinates ------------------
    sz = float(cfg.img_size)
    gt_boxes = (t_boxes01 * sz).cpu().numpy()
    pred_boxes = (p_boxes01 * sz).cpu().numpy()

    # ------------------ Denormalize images for visualization ------------------

    if norm_mean is None and norm_std is None:
        # use ImageNet values
        norm_mean = [0.485, 0.456, 0.406]
        norm_std = [0.229, 0.224, 0.225]

    imgs_denorm = denorm(
        [img for img in imgs],
        pers_norm=True,
        mean_ls=norm_mean,
        std_ls=norm_std,
    )  # list of tensors as input

    # Convert PIL -> numpy (BGR) for OpenCV-style drawing
    imgs_np = [np.array(img)[..., ::-1] for img in imgs_denorm]

    # ------------------ Prepare annotations ------------------
    gt_bboxes, gt_classes = [], []
    pred_bboxes, pred_classes = [], []

    for i in range(len(imgs)):
        gt_bboxes.append([tuple(map(int, gt_boxes[i]))])
        pred_bboxes.append([tuple(map(int, pred_boxes[i]))])

        # Map numeric class back to string
        gt_cls_id = int(t_cls[i].item())
        pred_cls_id = int(pred_cls[i].item())

        gt_classes.append([gt_cls_id])
        pred_classes.append([pred_cls_id])

    if annotations_dict is None:
        annotations_dict = {}

    # ------------------ Draw GT in green ------------------

    imgs_annot = draw_annotations(
        imgs_np,
        gt_bboxes,
        classes=gt_classes,
        colors=(0, 255, 0),
        prefix="",
        id2obj=LABEL_TO_CLASS,
        **annotations_dict,
    )

    # ------------------ Draw predictions in red ------------------
    imgs_annot = draw_annotations(
        imgs_annot,
        pred_bboxes,
        classes=pred_classes,
        colors=(0, 0, 255),
        prefix="pred:",
        id2obj=LABEL_TO_CLASS,
        **annotations_dict,
    )

    # ------------------ Compute IoU per image ------------------
    ious = []
    for i in range(len(imgs)):
        gt_box = torch.tensor(gt_boxes[i], dtype=torch.float32)
        pr_box = torch.tensor(pred_boxes[i], dtype=torch.float32)
        ious.append(iou(gt_box, pr_box).item())

    # ------------------ Show images in grid ------------------
    n_rows = int(np.ceil(len(imgs_annot) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.array(axes).reshape(-1)

    for i, ax in enumerate(axes):
        if i < len(imgs_annot):
            img_rgb = imgs_annot[i][..., ::-1]  # BGR -> RGB
            ax.imshow(img_rgb)
            ax.set_title(f"IoU={ious[i]:.2f}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()


## ------------ Optuna visualizations utilities -------------------- ##


def analyze_optuna_study(
    study, top_n=5, export_csv=True, csv_name="optuna_results.csv"
):
    """
    Analyze an Optuna study with visualizations and trial summaries.

    Args:
        study: Optuna study object (after optimization).
        top_n (int): Number of top trials to display.
        export_csv (bool): If True, export results to CSV.
        csv_name (str): Filename for the exported CSV.
    """

    # -------------------------------
    # Export results to CSV
    # -------------------------------
    df_trials = study.trials_dataframe()
    if export_csv:
        df_trials.to_csv(csv_name, index=False)
        print(f"[INFO] Results exported to {csv_name}")

    # -------------------------------
    # Show best trials
    # -------------------------------
    print("\n===== Top Trials =====")
    sorted_trials = sorted(
        study.trials,
        key=lambda t: (t.values[0], t.values[1]),  # sort by IoU first, then ACC
        reverse=True,
    )
    for t in sorted_trials[:top_n]:
        print(
            f"Trial {t.number} | IoU={t.values[0]:.4f}, ACC={t.values[1]:.4f}\n"
            f"Params: {t.params}\n"
        )

    # -------------------------------
    # Visualization plots
    # -------------------------------
    print("\n[INFO] Generating interactive plots...")

    # 1. Optimization history: track objective values across trials
    fig1 = plot_optimization_history(study)
    fig1.show()

    # # 2. Parameter importance (for IoU)
    # fig2 = plot_param_importances(
    #     study, target=lambda t: t.values[0], target_name="IoU"
    # )
    # fig2.show()

    # # 3. Parameter importance (for ACC)
    # fig3 = plot_param_importances(
    #     study, target=lambda t: t.values[1], target_name="ACC"
    # )
    # fig3.show()

    # # 4. Parallel coordinate plot: explore hyperparameter interactions
    # fig4 = plot_parallel_coordinate(
    #     study, target=lambda t: t.values[0], target_name="IoU"
    # )
    # fig4.show()

    # # 5. Slice plots: effect of each parameter on IoU
    # fig5 = plot_slice(study, target=lambda t: t.values[0], target_name="IoU")
    # fig5.show()

    # # 6. Contour plots: 2D interactions between parameters
    # fig6 = plot_contour(study, target=lambda t: t.values[0], target_name="IoU")
    # fig6.show()

    # # 7. Pareto front: trade-off between IoU and ACC (multi-objective)
    # fig7 = plot_pareto_front(study, target_names=["IoU", "ACC"])
    # fig7.show()

    print("[INFO] Analysis complete. Use the interactive plots to explore the results.")
