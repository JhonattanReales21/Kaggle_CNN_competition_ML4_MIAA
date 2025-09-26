import torch
import torch.nn.functional as F


## ---------- Metrics and Loss Functions ---------- ##


def iou(boxA, boxB):
    # box tensor [4] en píxeles
    xA = torch.max(boxA[..., 0], boxB[..., 0])
    yA = torch.max(boxA[..., 1], boxB[..., 1])
    xB = torch.min(boxA[..., 2], boxB[..., 2])
    yB = torch.min(boxA[..., 3], boxB[..., 3])
    inter = (xB - xA).clamp(min=0) * (yB - yA).clamp(min=0)
    areaA = (boxA[..., 2] - boxA[..., 0]).clamp(min=0) * (
        boxA[..., 3] - boxA[..., 1]
    ).clamp(min=0)
    areaB = (boxB[..., 2] - boxB[..., 0]).clamp(min=0) * (
        boxB[..., 3] - boxB[..., 1]
    ).clamp(min=0)
    union = areaA + areaB - inter + 1e-7
    return inter / union


def box_area(b):
    return ((b[..., 2] - b[..., 0]).clamp(min=0)) * (
        (b[..., 3] - b[..., 1]).clamp(min=0)
    )


def giou(pred, target, eps=1e-7):
    # pred, target en xyxy normalizado [0,1]
    x1 = torch.max(pred[..., 0], target[..., 0])
    y1 = torch.max(pred[..., 1], target[..., 1])
    x2 = torch.min(pred[..., 2], target[..., 2])
    y2 = torch.min(pred[..., 3], target[..., 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    ap = box_area(pred)
    at = box_area(target)
    union = ap + at - inter + eps
    iou_val = inter / union

    ex1 = torch.min(pred[..., 0], target[..., 0])
    ey1 = torch.min(pred[..., 1], target[..., 1])
    ex2 = torch.max(pred[..., 2], target[..., 2])
    ey2 = torch.max(pred[..., 3], target[..., 3])
    area_c = (ex2 - ex1).clamp(min=0) * (ey2 - ey1).clamp(min=0) + eps
    return iou_val - (area_c - union) / area_c


def giou_loss(pred, tgt):
    return (1.0 - giou(pred, tgt)).mean()


def smooth_l1_loss(input, target, beta=1.0):
    # PyTorch already has smooth_l1_loss, pero dejamos control de beta
    return F.smooth_l1_loss(input, target, beta=beta)


def diou(pred, target, eps=1e-7):  # pred/target en xyxy normalizado [0,1]
    x1 = torch.max(pred[..., 0], target[..., 0])
    y1 = torch.max(pred[..., 1], target[..., 1])
    x2 = torch.min(pred[..., 2], target[..., 2])
    y2 = torch.min(pred[..., 3], target[..., 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    ap = box_area(pred)
    at = box_area(target)
    union = ap + at - inter + eps
    iou_val = inter / union

    # distancia de centros (penaliza cajas lejos aun con IoU moderado)
    pcx = (pred[..., 0] + pred[..., 2]) / 2
    pcy = (pred[..., 1] + pred[..., 3]) / 2
    tcx = (target[..., 0] + target[..., 2]) / 2
    tcy = (target[..., 1] + target[..., 3]) / 2
    center_dist2 = (pcx - tcx) ** 2 + (pcy - tcy) ** 2

    # diagonal del rectángulo envolvente
    ex1 = torch.min(pred[..., 0], target[..., 0])
    ey1 = torch.min(pred[..., 1], target[..., 1])
    ex2 = torch.max(pred[..., 2], target[..., 2])
    ey2 = torch.max(pred[..., 3], target[..., 3])
    cw = (ex2 - ex1).clamp(min=0)
    ch = (ey2 - ey1).clamp(min=0)
    c2 = cw * cw + ch * ch + eps

    return iou_val - center_dist2 / c2


def diou_loss(pred, tgt):
    return (1.0 - diou(pred, tgt)).mean()
