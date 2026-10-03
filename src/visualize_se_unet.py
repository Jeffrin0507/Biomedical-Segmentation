import sys
from pathlib import Path

# Add project root to Python path
sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

import torch
import matplotlib.pyplot as plt
import numpy as np

from dataset import create_splits, KvasirDataset
from models.se_unet import SEUNet


# =========================
# Configuration
# =========================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

NUM_SAMPLES = 6

print("Using device:", DEVICE)


# =========================
# Dataset
# =========================

train_ids, val_ids, test_ids = create_splits()

test_dataset = KvasirDataset(
    test_ids,
    augment=False
)


# =========================
# Load SE-U-Net
# =========================

model = SEUNet(
    in_channels=3,
    out_channels=1
).to(DEVICE)

model.load_state_dict(
    torch.load(
        "results/se_unet_best.pth",
        map_location=DEVICE
    )
)

model.eval()

print("Best SE-U-Net model loaded.")


# =========================
# Random test samples
# =========================

indices = np.random.choice(
    len(test_dataset),
    NUM_SAMPLES,
    replace=False
)


# =========================
# Visualization
# =========================

for index in indices:

    image, mask = test_dataset[index]

    input_image = image.unsqueeze(0).to(DEVICE)

    # -------------------------
    # Prediction
    # -------------------------

    with torch.no_grad():

        output = model(input_image)

        probability = torch.sigmoid(output)

        prediction = (
            probability >= 0.5
        ).float()


    # -------------------------
    # Convert to NumPy
    # -------------------------

    image_np = image.cpu().numpy()

    mask_np = (
        mask.squeeze(0)
        .cpu()
        .numpy()
    )

    probability_np = (
        probability.squeeze()
        .cpu()
        .numpy()
    )

    prediction_np = (
        prediction.squeeze()
        .cpu()
        .numpy()
    )


    # -------------------------
    # Undo normalization
    # -------------------------

    image_np = image_np * 0.5 + 0.5

    image_np = np.transpose(
        image_np,
        (1, 2, 0)
    )

    image_np = np.clip(
        image_np,
        0,
        1
    )


    # -------------------------
    # Calculate Dice
    # -------------------------

    pred_flat = prediction_np.flatten()
    mask_flat = mask_np.flatten()

    TP = np.sum(
        (pred_flat == 1) &
        (mask_flat == 1)
    )

    FP = np.sum(
        (pred_flat == 1) &
        (mask_flat == 0)
    )

    FN = np.sum(
        (pred_flat == 0) &
        (mask_flat == 1)
    )

    dice = (
        2 * TP /
        (2 * TP + FP + FN + 1e-7)
    )


    # -------------------------
    # Probability information
    # -------------------------

    max_probability = probability_np.max()

    mean_positive_probability = (
        probability_np[mask_np == 1].mean()
        if np.any(mask_np == 1)
        else 0
    )


    # -------------------------
    # Plot
    # -------------------------

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(18, 5)
    )


    axes[0].imshow(image_np)

    axes[0].set_title(
        "Original Image"
    )

    axes[0].axis("off")


    axes[1].imshow(
        mask_np,
        cmap="gray"
    )

    axes[1].set_title(
        "Ground Truth"
    )

    axes[1].axis("off")


    axes[2].imshow(
        probability_np,
        cmap="viridis",
        vmin=0,
        vmax=1
    )

    axes[2].set_title(
        f"SE Probability Map\n"
        f"Max = {max_probability:.3f}"
    )

    axes[2].axis("off")


    axes[3].imshow(
        prediction_np,
        cmap="gray"
    )

    axes[3].set_title(
        f"SE-U-Net Prediction\n"
        f"Dice = {dice:.3f}"
    )

    axes[3].axis("off")


    plt.suptitle(
        f"SE-U-Net Test Sample {index} | "
        f"Mean probability inside GT = "
        f"{mean_positive_probability:.3f}"
    )

    plt.tight_layout()

    plt.show()