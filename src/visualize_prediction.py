import sys
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import DataLoader

from dataset import create_splits, KvasirDataset
from models.unet import UNet


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
# Load model
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

    # Add batch dimension
    input_image = image.unsqueeze(0).to(DEVICE)

    # Prediction
    with torch.no_grad():

        output = model(input_image)

        probability = torch.sigmoid(output)

        prediction = (
            probability >= 0.5
        ).float()


    # Move to CPU
    image_np = image.cpu().numpy()
    mask_np = mask.squeeze(0).cpu().numpy()
    prediction_np = prediction.squeeze().cpu().numpy()


    # Undo image normalization
    image_np = image_np * 0.5 + 0.5

    # CHW -> HWC
    image_np = np.transpose(
        image_np,
        (1, 2, 0)
    )

    # Keep values valid
    image_np = np.clip(
        image_np,
        0,
        1
    )


    # =========================
    # Plot
    # =========================

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5)
    )


    axes[0].imshow(image_np)
    axes[0].set_title("Original Image")
    axes[0].axis("off")


    axes[1].imshow(mask_np, cmap="gray")
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")


    axes[2].imshow(prediction_np, cmap="gray")
    axes[2].set_title("U-Net Prediction")
    axes[2].axis("off")


    plt.tight_layout()
    plt.show()