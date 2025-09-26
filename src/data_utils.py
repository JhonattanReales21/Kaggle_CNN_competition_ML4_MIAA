from torch.utils.data import Dataset, DataLoader

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
