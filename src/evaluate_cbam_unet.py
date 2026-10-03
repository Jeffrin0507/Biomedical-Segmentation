import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

import torch
import numpy as np
from torch.utils.data import DataLoader

from dataset import create_splits, KvasirDataset
from models.cbam_unet import CBAMUNet


# ============================================================
# Configuration
# ============================================================

BATCH_SIZE = 8

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Using device:", DEVICE)


# ============================================================
# Dataset
# ============================================================

train_ids, val_ids, test_ids = create_splits()

test_dataset = KvasirDataset(
    test_ids,
    augment=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Test samples:", len(test_dataset))


# ============================================================
# Model
# ============================================================

model = CBAMUNet(
    in_channels=3,
    out_channels=1
).to(DEVICE)

model.load_state_dict(
    torch.load(
        "results/cbam_unet_best.pth",
        map_location=DEVICE
    )
)

model.eval()

print("Best CBAM-U-Net model loaded.")


# ============================================================
# Metric storage
# ============================================================

dice_scores = []
iou_scores = []
precision_scores = []
recall_scores = []


# ============================================================
# Evaluation
# ============================================================

with torch.no_grad():

    for images, masks in test_loader:

        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        outputs = model(images)

        probabilities = torch.sigmoid(outputs)

        predictions = (
            probabilities >= 0.5
        ).float()

        # Process each image separately
        for pred, target in zip(
            predictions,
            masks
        ):

            pred = pred.squeeze(0)
            target = target.squeeze(0)

            pred_flat = pred.flatten()
            target_flat = target.flatten()

            TP = torch.sum(
                (pred_flat == 1) &
                (target_flat == 1)
            ).item()

            FP = torch.sum(
                (pred_flat == 1) &
                (target_flat == 0)
            ).item()

            FN = torch.sum(
                (pred_flat == 0) &
                (target_flat == 1)
            ).item()

            dice = (
                2 * TP /
                (2 * TP + FP + FN + 1e-7)
            )

            iou = (
                TP /
                (TP + FP + FN + 1e-7)
            )

            precision = (
                TP /
                (TP + FP + 1e-7)
            )

            recall = (
                TP /
                (TP + FN + 1e-7)
            )

            dice_scores.append(dice)
            iou_scores.append(iou)
            precision_scores.append(precision)
            recall_scores.append(recall)


# ============================================================
# Final results
# ============================================================

mean_dice = np.mean(dice_scores)
mean_iou = np.mean(iou_scores)
mean_precision = np.mean(precision_scores)
mean_recall = np.mean(recall_scores)


print()
print("=" * 50)
print("CBAM-U-Net Test Results")
print("=" * 50)

print(
    f"Dice      : {mean_dice:.4f}"
)

print(
    f"IoU       : {mean_iou:.4f}"
)

print(
    f"Precision : {mean_precision:.4f}"
)

print(
    f"Recall    : {mean_recall:.4f}"
)

print("=" * 50)