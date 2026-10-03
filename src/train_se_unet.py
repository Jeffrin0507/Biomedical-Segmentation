import sys
from pathlib import Path

# Add project root to Python path
sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from dataset import create_splits, KvasirDataset
from losses import combined_loss

from models.se_unet import SEUNet


# =========================
# Configuration
# =========================

EPOCHS = 20
LEARNING_RATE = 1e-4
BATCH_SIZE = 8

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", DEVICE)


# =========================
# Dataset
# =========================

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
    num_workers=0,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)


# =========================
# Model
# =========================

model = SEUNet(
    in_channels=3,
    out_channels=1
).to(DEVICE)


# =========================
# Optimizer
# =========================

optimizer = Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# =========================
# Training
# =========================

best_val_loss = float("inf")

train_history = []
val_history = []


for epoch in range(EPOCHS):

    # ---------------------
    # Training
    # ---------------------

    model.train()

    running_train_loss = 0.0

    for images, masks in train_loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        masks = masks.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad()

        outputs = model(images)

        loss = combined_loss(
            outputs,
            masks
        )

        loss.backward()

        optimizer.step()

        running_train_loss += loss.item()


    train_loss = (
        running_train_loss /
        len(train_loader)
    )


    # ---------------------
    # Validation
    # ---------------------

    model.eval()

    running_val_loss = 0.0

    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            masks = masks.to(
                DEVICE,
                non_blocking=True
            )

            outputs = model(images)

            loss = combined_loss(
                outputs,
                masks
            )

            running_val_loss += loss.item()


    val_loss = (
        running_val_loss /
        len(val_loader)
    )


    # ---------------------
    # Save history
    # ---------------------

    train_history.append(train_loss)
    val_history.append(val_loss)


    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} "
        f"Val Loss: {val_loss:.4f}"
    )


    # ---------------------
    # Save best model
    # ---------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            "results/se_unet_best.pth"
        )

        print(
            f"  --> Best SE-U-Net saved "
            f"(Val Loss: {best_val_loss:.4f})"
        )


# =========================
# Save final model
# =========================

torch.save(
    model.state_dict(),
    "results/se_unet_final.pth"
)


# =========================
# Save history
# =========================

history = {
    "train_loss": train_history,
    "val_loss": val_history
}

torch.save(
    history,
    "results/se_unet_history.pth"
)


# =========================
# Final information
# =========================

print("\nTraining complete.")

print(
    "Best validation loss:",
    best_val_loss
)

print(
    "Best model saved to:",
    "results/se_unet_best.pth"
)

print(
    "Final model saved to:",
    "results/se_unet_final.pth"
)