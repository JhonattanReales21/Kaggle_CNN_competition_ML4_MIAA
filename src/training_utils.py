import torch
import torch.nn.functional as F
import torch.nn as nn


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
