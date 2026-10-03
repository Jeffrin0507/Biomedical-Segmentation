import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm

from dataset import create_splits, KvasirDataset
from losses import combined_loss
from models.cbam_unet import CBAMUNet


# ============================================================
# Configuration
# ============================================================

EPOCHS = 20
LEARNING_RATE = 1e-4
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

train_dataset = KvasirDataset(
    train_ids,
    augment=True
)

val_dataset = KvasirDataset(
    val_ids,
    augment=False
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print()
print("Dataset split:")
print("Training   :", len(train_dataset))
print("Validation :", len(val_dataset))
print("Testing    :", len(test_ids))


# ============================================================
# Model
# ============================================================

model = CBAMUNet(
    in_channels=3,
    out_channels=1
).to(DEVICE)

print()
print("CBAM-U-Net created.")

# ============================================================
# Optimizer
# ============================================================

optimizer = Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# Training
# ============================================================

best_val_loss = float("inf")

history = {
    "train_loss": [],
    "val_loss": []
}


for epoch in range(EPOCHS):

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    model.train()

    train_loss = 0.0

    train_progress = tqdm(
        train_loader,
        desc=f"Epoch [{epoch + 1}/{EPOCHS}]"
    )

    for images, masks in train_progress:

        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = combined_loss(
            outputs,
            masks
        )

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

        train_progress.set_postfix(
            loss=loss.item()
        )

    train_loss /= len(train_loader)

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    model.eval()

    val_loss = 0.0

    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            outputs = model(images)

            loss = combined_loss(
                outputs,
                masks
            )

            val_loss += loss.item()

    val_loss /= len(val_loader)

    # --------------------------------------------------------
    # Save history
    # --------------------------------------------------------

    history["train_loss"].append(
        train_loss
    )

    history["val_loss"].append(
        val_loss
    )

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} "
        f"Val Loss: {val_loss:.4f}"
    )

    # --------------------------------------------------------
    # Save best model
    # --------------------------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            "results/cbam_unet_best.pth"
        )

        print(
            "  -> Best CBAM-U-Net model saved."
        )


# ============================================================
# Save final model and history
# ============================================================

torch.save(
    model.state_dict(),
    "results/cbam_unet_final.pth"
)

torch.save(
    history,
    "results/cbam_unet_history.pth"
)

print()
print("Training complete.")
print(
    "Best validation loss:",
    best_val_loss
)