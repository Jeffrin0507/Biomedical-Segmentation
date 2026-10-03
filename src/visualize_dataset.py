from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


# Dataset location
DATASET_DIR = Path(
    "dataset/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG"
)

IMAGE_DIR = DATASET_DIR / "images"
MASK_DIR = DATASET_DIR / "masks"


# Select one image
image_path = sorted(IMAGE_DIR.glob("*.jpg"))[0]

# Find corresponding mask
mask_path = MASK_DIR / image_path.name


# Load image and mask
image = Image.open(image_path)
mask = Image.open(mask_path)


# Print information
print("Image:", image_path.name)
print("Image size:", image.size)
print("Mask size:", mask.size)
print("Mask mode:", mask.mode)


# Display
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original Endoscopy Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(mask, cmap="gray")
plt.title("Ground Truth Mask")
plt.axis("off")

plt.tight_layout()
plt.show()