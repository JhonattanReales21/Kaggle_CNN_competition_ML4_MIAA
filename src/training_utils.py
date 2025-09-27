import torch
import torch.nn.functional as F
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

import os
import sys

sys.path.append("..")

from src.data_utils import *
from src.cnn_models import *

## ---------- Metrics and Loss Functions ---------- ##


def iou(boxA, boxB, eps=1e-7):
    """
    Compute IoU between two boxes or batches of boxes.
    boxA, boxB: tensors [..., 4] in [x1, y1, x2, y2] format (absolute or normalized).
    """
    # Intersection coords
    x1 = torch.max(boxA[..., 0], boxB[..., 0])
    y1 = torch.max(boxA[..., 1], boxB[..., 1])
    x2 = torch.min(boxA[..., 2], boxB[..., 2])
    y2 = torch.min(boxA[..., 3], boxB[..., 3])

    # Intersection area
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

    # Areas
    areaA = (boxA[..., 2] - boxA[..., 0]).clamp(min=0) * (
        boxA[..., 3] - boxA[..., 1]
    ).clamp(min=0)
    areaB = (boxB[..., 2] - boxB[..., 0]).clamp(min=0) * (
        boxB[..., 3] - boxB[..., 1]
    ).clamp(min=0)

    # IoU
    return inter / (areaA + areaB - inter + eps)


def box_area(b):
    return ((b[..., 2] - b[..., 0]).clamp(min=0)) * (
        (b[..., 3] - b[..., 1]).clamp(min=0)
    )


def giou(pred, target, eps=1e-7):
    """
    Compute Generalized IoU (GIoU) between predicted and target boxes.
    Args:
        pred, target: tensors [..., 4] in [x1, y1, x2, y2] normalized [0,1].
    Returns:
        GIoU value(s).
    """
    # Intersection
    x1 = torch.max(pred[..., 0], target[..., 0])
    y1 = torch.max(pred[..., 1], target[..., 1])
    x2 = torch.min(pred[..., 2], target[..., 2])
    y2 = torch.min(pred[..., 3], target[..., 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

    # Areas
    area_p = box_area(pred)
    area_t = box_area(target)
    union = area_p + area_t - inter + eps
    iou_val = inter / union

    # Smallest enclosing box
    ex1 = torch.min(pred[..., 0], target[..., 0])
    ey1 = torch.min(pred[..., 1], target[..., 1])
    ex2 = torch.max(pred[..., 2], target[..., 2])
    ey2 = torch.max(pred[..., 3], target[..., 3])
    area_c = (ex2 - ex1).clamp(min=0) * (ey2 - ey1).clamp(min=0) + eps

    # GIoU = IoU - (C - Union) / C
    return iou_val - (area_c - union) / area_c


def giou_loss(pred, target):
    """
    Generalized IoU loss.
    Args:
        pred, target: tensors [..., 4] in [x1, y1, x2, y2].
    Returns:
        Scalar loss (1 - GIoU).
    """
    return (1.0 - giou(pred, target)).mean()


def smooth_l1_loss(input, target, beta=1.0):
    # PyTorch already has smooth_l1_loss, just add beta parameter
    return F.smooth_l1_loss(input, target, beta=beta)


def diou(pred, target, eps=1e-7):
    """
    Compute Distance IoU (DIoU) between predicted and target boxes.
    Args:
        pred, target: tensors [..., 4] in [x1, y1, x2, y2] normalized [0,1].
    Returns:
        DIoU value(s).
    """
    # Intersection
    x1 = torch.max(pred[..., 0], target[..., 0])
    y1 = torch.max(pred[..., 1], target[..., 1])
    x2 = torch.min(pred[..., 2], target[..., 2])
    y2 = torch.min(pred[..., 3], target[..., 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

    # Areas and IoU
    area_p = box_area(pred)
    area_t = box_area(target)
    union = area_p + area_t - inter + eps
    iou_val = inter / union

    # Centers of both boxes
    pcx = (pred[..., 0] + pred[..., 2]) / 2
    pcy = (pred[..., 1] + pred[..., 3]) / 2
    tcx = (target[..., 0] + target[..., 2]) / 2
    tcy = (target[..., 1] + target[..., 3]) / 2
    center_dist2 = (pcx - tcx) ** 2 + (pcy - tcy) ** 2

    # Diagonal length of smallest enclosing box
    ex1 = torch.min(pred[..., 0], target[..., 0])
    ey1 = torch.min(pred[..., 1], target[..., 1])
    ex2 = torch.max(pred[..., 2], target[..., 2])
    ey2 = torch.max(pred[..., 3], target[..., 3])
    cw = (ex2 - ex1).clamp(min=0)
    ch = (ey2 - ey1).clamp(min=0)
    c2 = cw * cw + ch * ch + eps

    # DIoU = IoU - (distance^2 / diagonal^2)
    return iou_val - center_dist2 / c2


def diou_loss(pred, target):
    """
    Distance IoU loss.
    Args:
        pred, target: tensors [..., 4] in [x1, y1, x2, y2].
    Returns:
        Scalar loss (1 - DIoU).
    """
    return (1.0 - diou(pred, target)).mean()


## ---------- Training Utilities ---------- ##


class Trainer_base:
    """
    Base trainer class for CNN models with classification and bounding box regression.

    Responsibilities:
    - Run training and validation loops.
    - Compute classification and regression losses.
    - Monitor metrics (loss, IoU, accuracy, or combined score).
    - Handle optimizer, scheduler, gradient clipping, and early stopping.
    - Save the best models according to IoU or monitored metric.
    """

    def __init__(
        self,
        model,
        dl_train,
        dl_valid,
        cfg,
        optimizer=None,
        scheduler=None,
        device="cuda",
    ):
        """
        Initialize the Trainer.

        Args:
            model (nn.Module): The CNN model (with classification and bbox heads).
            dl_train (DataLoader): PyTorch DataLoader for training data.
            dl_valid (DataLoader): PyTorch DataLoader for validation data.
            cfg (Namespace or dict): Configuration object with hyperparameters
                (epochs, lr, weight_decay, cls_loss_w, box_loss_w, beta_smoothl1, etc.).
            optimizer (torch.optim.Optimizer, optional): Optimizer for training.
                If None, AdamW will be used with cfg.lr and cfg.weight_decay.
            scheduler (torch.optim.lr_scheduler, optional): LR scheduler.
            device (str): Training device ("cuda" or "cpu").
        """
        self.model = model.to(device)
        self.dl_train = dl_train
        self.dl_valid = dl_valid
        self.cfg = cfg
        self.device = device

        # Best IoU seen during training (for saving checkpoints)
        self._best_iou = -1.0

        # Optimizer
        self.opt = optimizer or torch.optim.AdamW(
            self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
        )

        # Scheduler (optional)
        self.sched = scheduler

    # --------------------------------------------------------
    def _monitor_value(self, metrics: dict):
        """
        Returns the value to monitor according to cfg.monitor:
        - "val_loss": lower is better
        - "val_iou": higher is better
        - "val_score": alpha*IoU + (1-alpha)*ACC
        """
        monitor = getattr(self.cfg, "monitor", "val_loss")
        if monitor == "val_iou":
            return metrics["mean_iou"]
        elif monitor == "val_score":
            alpha = getattr(self.cfg, "alpha_score", 0.6)
            return alpha * metrics["mean_iou"] + (1.0 - alpha) * metrics["acc"]
        else:  # default: val_loss
            return metrics["loss"]

    # --------------------------------------------------------
    def _step_batch(self, batch, train: bool = True):
        """
        Run a single batch through the model, compute losses and metrics.

        Args:
            batch: (imgs, target_boxes, target_cls, filenames)
            train (bool): If True, do backward pass and optimizer step.

        Returns:
            loss, cls_loss, box_loss, mean_iou (floats)
        """
        imgs, t_boxes01, t_cls, _ = batch
        imgs, t_boxes01, t_cls = (
            imgs.to(self.device),
            t_boxes01.to(self.device),
            t_cls.to(self.device),
        )

        if train:
            self.opt.zero_grad(set_to_none=True)

        logits, pred_box01 = self.model(imgs)

        # Ensure class targets have correct shape
        t_cls = t_cls.squeeze(1) if t_cls.ndim > 1 else t_cls

        # Classification loss (with pos_weight if available)
        cls_loss = F.binary_cross_entropy_with_logits(
            logits, t_cls, pos_weight=getattr(self.cfg, "positive_weight", None)
        )

        # Box regression loss (SmoothL1 + optional DIoU)
        l1 = F.smooth_l1_loss(pred_box01, t_boxes01, beta=self.cfg.beta_smoothl1)
        if "diou_loss" in globals():
            d = diou_loss(pred_box01, t_boxes01)
            box_loss = 0.5 * l1 + 0.5 * d
        else:
            box_loss = l1

        # Weighted total loss
        loss = self.cfg.cls_loss_w * cls_loss + self.cfg.box_loss_w * box_loss

        if train:
            loss.backward()
            # Gradient clipping
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.opt.step()

        # ----- Metrics (no gradient) -----
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            self.cfg.acc_metric.update(probs, t_cls)

            # IoU in pixels (per sample, averaged)
            sz = float(self.cfg.img_size)
            pred_px = pred_box01 * sz
            tgt_px = t_boxes01 * sz
            ious = [(iou(pred_px[i], tgt_px[i]).item()) for i in range(len(pred_px))]
            mean_iou = float(np.mean(ious)) if ious else 0.0

        return (
            float(loss.item()),
            float(cls_loss.item()),
            float(box_loss.item()),
            mean_iou,
        )

    # --------------------------------------------------------
    def _run_epoch(self, train: bool = True):
        """
        Run one full epoch (train or validation).

        Args:
            train (bool): If True, training mode; else validation.

        Returns:
            dict with averaged metrics: loss, cls_loss, box_loss, mean_iou, acc
        """
        loader = self.dl_train if train else self.dl_valid
        self.model.train(train)

        total_loss = total_cls_loss = total_box_loss = iou_sum = 0.0
        n_batches = 0

        # Reset accuracy metric at the start of the epoch
        self.cfg.acc_metric.reset()

        for batch in loader:
            loss, cls_l, box_l, miou = self._step_batch(batch, train=train)
            total_loss += loss
            total_cls_loss += cls_l
            total_box_loss += box_l
            iou_sum += miou
            n_batches += 1

        acc = self.cfg.acc_metric.compute().item() if n_batches > 0 else 0.0

        return {
            "loss": total_loss / max(1, n_batches),
            "cls_loss": total_cls_loss / max(1, n_batches),
            "box_loss": total_box_loss / max(1, n_batches),
            "mean_iou": iou_sum / max(1, n_batches),
            "acc": acc,
        }

    # --------------------------------------------------------
    def fit(self):
        """
        Main training loop with early stopping.
        """
        monitor = getattr(self.cfg, "monitor", "val_loss")
        best_val = float("inf") if monitor == "val_loss" else -float("inf")
        es_patience = getattr(self.cfg, "es_patience", 4)
        wait = 0
        history = []

        for epoch in range(1, self.cfg.epochs + 1):
            # Training and validation
            tr = self._run_epoch(train=True)
            vl = self._run_epoch(train=False)

            # LR scheduler step
            if self.sched is not None:
                self.sched.step()

            # Log history
            history.append(
                {
                    "epoch": epoch,
                    "train": tr,
                    "valid": vl,
                    "lr": self.opt.param_groups[0]["lr"],
                }
            )

            # Always track best IoU
            if vl["mean_iou"] > self._best_iou:
                self._best_iou = vl["mean_iou"]
                self.save_model(
                    f"best_val_iou_model.pt", output_dir=self.cfg.output_dir
                )

            # Check monitored metric
            monitor_value = self._monitor_value(vl)
            better = (
                (monitor_value < best_val)
                if monitor == "val_loss"
                else (monitor_value > best_val)
            )

            # Logging, print every n epochs
            if (epoch == 1 or epoch == self.cfg.epochs) or (
                epoch % self.cfg.logging_step == 0
            ):
                print(
                    f"Epoch {epoch:03d} | LR {self.opt.param_groups[0]['lr']:.2e} | "
                    f"Train -- loss: {tr['loss']:.4f}, acc: {tr['acc']:.3f}, IoU: {tr['mean_iou']:.3f} | "
                    f"Valid -- loss: {vl['loss']:.4f}, acc: {vl['acc']:.3f}, IoU: {vl['mean_iou']:.3f} | "
                    f"Monitor({monitor}): {monitor_value:.4f}"
                )

            # Early stopping check
            if better:
                best_val = monitor_value
                wait = 0
                self.save_model(
                    f"best_monitor_model.pt", output_dir=self.cfg.output_dir
                )
            else:
                wait += 1
                if wait >= es_patience:
                    print(f"\nEarly stopping at epoch {epoch}:\n")
                    print(
                        f"Best {monitor}: {best_val:.4f}\n"
                        f"Train: loss: {tr['loss']:.4f}, acc: {tr['acc']:.3f}, IoU: {tr['mean_iou']:.3f}\n"
                        f"Valid: loss: {vl['loss']:.4f}, acc: {vl['acc']:.3f}, IoU: {vl['mean_iou']:.3f}"
                    )
                    break

        return history

    # --------------------------------------------------------

    def save_model(self, path: str, output_dir: str = "outputs"):
        """
        Save the model checkpoint (state dict + config) into a structured folder.

        Structure:
            output_dir/
                cnn_name/
                    path  (e.g., best_model.pt)

        Args:
            path (str): Filename for the checkpoint (e.g., "best.pt").
            output_dir (str): Base folder to store all checkpoints. Default = "outputs".
        """
        # Build the directory: output_dir/<cnn_name>/
        model_dir = os.path.join(output_dir, self.cfg.cnn_name)
        os.makedirs(model_dir, exist_ok=True)

        # Full save path
        save_path = os.path.join(model_dir, path)

        # Save checkpoint: config + model weights
        torch.save(
            {"cfg": self.cfg.__dict__, "model_state": self.model.state_dict()},
            save_path,
        )
