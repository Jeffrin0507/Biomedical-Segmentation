import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

import torch
import matplotlib.pyplot as plt
import numpy as np

from dataset import create_splits, KvasirDataset
from models.unet import UNet
from models.se_unet import SEUNet
from models.cbam_unet import CBAMUNet


# ============================================================
# Configuration
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

# Fixed test samples
# We use indices instead of random selection so that
# the same samples can be reproduced later.
TEST_INDICES = [35, 32, 68, 107, 160]

print("Using device:", DEVICE)


# ============================================================
# Dataset
# ============================================================

train_ids, val_ids, test_ids = create_splits()

test_dataset = KvasirDataset(
    test_ids,
    augment=False
)

print("Test samples:", len(test_dataset))


# ============================================================
# Load Models
# ============================================================

unet = UNet(
    in_channels=3,
    out_channels=1
).to(DEVICE)

se_unet = SEUNet(
    in_channels=3,
    out_channels=1
).to(DEVICE)

cbam_unet = CBAMUNet(
    in_channels=3,
    out_channels=1
).to(DEVICE)


unet.load_state_dict(
    torch.load(
        "results/unet_best.pth",
        map_location=DEVICE
    )
)

se_unet.load_state_dict(
    torch.load(
        "results/se_unet_best.pth",
        map_location=DEVICE
    )
)

cbam_unet.load_state_dict(
    torch.load(
        "results/cbam_unet_best.pth",
        map_location=DEVICE
    )
)


unet.eval()
se_unet.eval()
cbam_unet.eval()

print("All three models loaded.")


# ============================================================
# Dice Function
# ============================================================

def calculate_dice(prediction, target):

    prediction = prediction.flatten()
    target = target.flatten()

    TP = np.sum(
        (prediction == 1) &
        (target == 1)
    )

    FP = np.sum(
        (prediction == 1) &
        (target == 0)
    )

    FN = np.sum(
        (prediction == 0) &
        (target == 1)
    )

    dice = (
        2 * TP /
        (2 * TP + FP + FN + 1e-7)
    )

    return dice


# ============================================================
# Prediction Function
# ============================================================

def get_prediction(model, image):

    input_image = image.unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        output = model(input_image)

        probability = torch.sigmoid(output)

        prediction = (
            probability >= 0.5
        ).float()

    return (
        probability.squeeze().cpu().numpy(),
        prediction.squeeze().cpu().numpy()
    )


# ============================================================
# Compare Models
# ============================================================

for index in TEST_INDICES:

    image, mask = test_dataset[index]

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    mask_np = (
        mask.squeeze(0)
        .cpu()
        .numpy()
    )

    # --------------------------------------------------------
    # U-Net
    # --------------------------------------------------------

    unet_probability, unet_prediction = get_prediction(
        unet,
        image
    )

    # --------------------------------------------------------
    # SE-U-Net
    # --------------------------------------------------------

    se_probability, se_prediction = get_prediction(
        se_unet,
        image
    )

    # --------------------------------------------------------
    # CBAM-U-Net
    # --------------------------------------------------------

    cbam_probability, cbam_prediction = get_prediction(
        cbam_unet,
        image
    )

    # --------------------------------------------------------
    # Dice scores
    # --------------------------------------------------------

    unet_dice = calculate_dice(
        unet_prediction,
        mask_np
    )

    se_dice = calculate_dice(
        se_prediction,
        mask_np
    )

    cbam_dice = calculate_dice(
        cbam_prediction,
        mask_np
    )

    # --------------------------------------------------------
    # Recover original image
    # --------------------------------------------------------

    image_np = (
        image.cpu().numpy() * 0.5 + 0.5
    )

    image_np = np.transpose(
        image_np,
        (1, 2, 0)
    )

    image_np = np.clip(
        image_np,
        0,
        1
    )

    # ========================================================
    # Create figure
    # ========================================================

    fig, axes = plt.subplots(
        1,
        5,
        figsize=(22, 5)
    )

    # --------------------------------------------------------
    # Original
    # --------------------------------------------------------

    axes[0].imshow(image_np)

    axes[0].set_title(
        "Original Image"
    )

    axes[0].axis("off")

    # --------------------------------------------------------
    # Ground Truth
    # --------------------------------------------------------

    axes[1].imshow(
        mask_np,
        cmap="gray"
    )

    axes[1].set_title(
        "Ground Truth"
    )

    axes[1].axis("off")

    # --------------------------------------------------------
    # U-Net
    # --------------------------------------------------------

    axes[2].imshow(
        unet_prediction,
        cmap="gray"
    )

    axes[2].set_title(
        f"U-Net\nDice = {unet_dice:.3f}"
    )

    axes[2].axis("off")

    # --------------------------------------------------------
    # SE-U-Net
    # --------------------------------------------------------

    axes[3].imshow(
        se_prediction,
        cmap="gray"
    )

    axes[3].set_title(
        f"SE-U-Net\nDice = {se_dice:.3f}"
    )

    axes[3].axis("off")

    # --------------------------------------------------------
    # CBAM-U-Net
    # --------------------------------------------------------

    axes[4].imshow(
        cbam_prediction,
        cmap="gray"
    )

    axes[4].set_title(
        f"CBAM-U-Net\nDice = {cbam_dice:.3f}"
    )

    axes[4].axis("off")

    # --------------------------------------------------------
    # Overall title
    # --------------------------------------------------------

    plt.suptitle(
        f"Same Test Image Comparison | Test Sample {index}",
        fontsize=15
    )

    plt.tight_layout()

    plt.show()

    # ========================================================
    # Print numerical comparison
    # ========================================================

    print()
    print("=" * 60)
    print(f"Test Sample {index}")
    print("=" * 60)

    print(
        f"U-Net       Dice: {unet_dice:.4f}"
    )

    print(
        f"SE-U-Net    Dice: {se_dice:.4f}"
    )

    print(
        f"CBAM-U-Net  Dice: {cbam_dice:.4f}"
    )