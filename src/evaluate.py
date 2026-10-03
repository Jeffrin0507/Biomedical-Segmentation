import sys
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import create_splits, KvasirDataset
from models.unet import UNet


# =========================
# Configuration
# =========================

BATCH_SIZE = 8

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", DEVICE)


# =========================
# Dataset
# =========================

train_ids, val_ids, test_ids = create_splits()

test_dataset = KvasirDataset(
    test_ids,
    augment=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

print("Test samples:", len(test_dataset))


# =========================
# Load best U-Net
# =========================

model = UNet(
    in_channels=3,
    out_channels=1
).to(DEVICE)

model.load_state_dict(
    torch.load(
        "results/unet_best.pth",
        map_location=DEVICE
    )
)

model.eval()

print("Best U-Net model loaded.")


# =========================
# Sample-wise metric function
# =========================

def calculate_metrics(prediction, target):

    prediction = prediction.view(-1)
    target = target.view(-1)

    TP = ((prediction == 1) & (target == 1)).sum().item()
    FP = ((prediction == 1) & (target == 0)).sum().item()
    FN = ((prediction == 0) & (target == 1)).sum().item()

    epsilon = 1e-7

    dice = (
        2 * TP
        /
        (2 * TP + FP + FN + epsilon)
    )

    iou = (
        TP
        /
        (TP + FP + FN + epsilon)
    )

    precision = (
        TP
        /
        (TP + FP + epsilon)
    )

    recall = (
        TP
        /
        (TP + FN + epsilon)
    )

    return dice, iou, precision, recall


# =========================
# Evaluation
# =========================

all_dice = []
all_iou = []
all_precision = []
all_recall = []


with torch.no_grad():

    for images, masks in tqdm(
        test_loader,
        desc="Evaluating"
    ):

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        masks = masks.to(
            DEVICE,
            non_blocking=True
        )

        # Model prediction
        outputs = model(images)

        # Logits -> probability
        probabilities = torch.sigmoid(outputs)

        # Probability -> binary mask
        predictions = (
            probabilities >= 0.5
        ).float()


        # -------------------------
        # Calculate each sample
        # individually
        # -------------------------

        for i in range(images.size(0)):

            dice, iou, precision, recall = calculate_metrics(
                predictions[i],
                masks[i]
            )

            all_dice.append(dice)
            all_iou.append(iou)
            all_precision.append(precision)
            all_recall.append(recall)


# =========================
# Final sample-wise averages
# =========================

avg_dice = sum(all_dice) / len(all_dice)
avg_iou = sum(all_iou) / len(all_iou)
avg_precision = sum(all_precision) / len(all_precision)
avg_recall = sum(all_recall) / len(all_recall)


# =========================
# Results
# =========================

print("\n================================")
print("      U-NET TEST RESULTS")
print("     SAMPLE-WISE AVERAGE")
print("================================")

print(f"Test Samples : {len(all_dice)}")
print(f"Dice Score   : {avg_dice:.4f}")
print(f"IoU          : {avg_iou:.4f}")
print(f"Precision    : {avg_precision:.4f}")
print(f"Recall       : {avg_recall:.4f}")

print("================================")
