import torch
import torch.nn.functional as F
import torch.nn as nn


## ---------- Net architecture (Decision head) ---------- ##


class TwoHeadNetVOC(nn.Module):
    """Two-headed network for classification and regression of VOC bboxes.

    Args:
        nn (Module): PyTorch neural network module.
    """

    def __init__(self, backbone: nn.Module, feat_dim: int = 512, dropout: float = 0.2):
        super().__init__()
        self.backbone = backbone

        # Classification head
        self.cls_head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),  # 1 logit (binary classification)
        )

        # Box regression head -> directly VOC format [x1,y1,x2,y2] normalized
        self.box_head = nn.Sequential(
            nn.Linear(feat_dim, 256), nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, 4)
        )

    def forward(self, x):
        feats = self.backbone(x)
        logits = self.cls_head(feats).squeeze(1)
        boxes = self.box_head(feats)
        return logits, boxes


## ---------- Net architecture (Personalized Backbone) ---------- ##


class BaseBackbone(nn.Module):
    def __init__(self, base_ch=32, feat_dim=512):
        super().__init__()
        c = base_ch
        self.body = nn.Sequential(
            nn.Conv2d(3, c, 3, 2, 1),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, 1, 1),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, 2 * c, 3, 2, 1),
            nn.BatchNorm2d(2 * c),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * c, 2 * c, 3, 1, 1),
            nn.BatchNorm2d(2 * c),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * c, 4 * c, 3, 2, 1),
            nn.BatchNorm2d(4 * c),
            nn.ReLU(inplace=True),
            nn.Conv2d(4 * c, 4 * c, 3, 1, 1),
            nn.BatchNorm2d(4 * c),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(4 * c, feat_dim)

    def forward(self, x):
        x = self.body(x).flatten(1)
        return self.proj(x)
