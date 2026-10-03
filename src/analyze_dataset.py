from pathlib import Path
from PIL import Image
import numpy as np


DATASET_DIR = Path(
    "dataset/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG"
)

IMAGE_DIR = DATASET_DIR / "images"
MASK_DIR = DATASET_DIR / "masks"


image_files = sorted(IMAGE_DIR.glob("*.jpg"))

widths = []
heights = []
mask_values = set()

mismatched = []


for image_path in image_files:

    mask_path = MASK_DIR / image_path.name

    image = Image.open(image_path)
    mask = Image.open(mask_path)

    width, height = image.size

    widths.append(width)
    heights.append(height)

    # Check image/mask dimensions
    if image.size != mask.size:
        mismatched.append(image_path.name)

    # Get unique mask values
    mask_array = np.array(mask)

    for value in np.unique(mask_array):
        mask_values.add(int(value))


print("========== DATASET ANALYSIS ==========")

print(f"Number of images: {len(image_files)}")

print(f"\nMinimum width : {min(widths)}")
print(f"Maximum width : {max(widths)}")

print(f"Minimum height: {min(heights)}")
print(f"Maximum height: {max(heights)}")

print("\nMask pixel values:")
print(sorted(mask_values))

print("\nDimension mismatches:")
print(len(mismatched))

if mismatched:
    print("First few mismatches:")
    for name in mismatched[:10]:
        print(name)