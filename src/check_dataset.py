from pathlib import Path

# Kvasir-SEG dataset location
DATASET_DIR = Path("dataset/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG")

IMAGE_DIR = DATASET_DIR / "images"
MASK_DIR = DATASET_DIR / "masks"

image_files = sorted(IMAGE_DIR.glob("*"))
mask_files = sorted(MASK_DIR.glob("*"))

print("========== Kvasir-SEG Dataset ==========")

print(f"Number of images : {len(image_files)}")
print(f"Number of masks  : {len(mask_files)}")

print("\nFirst 5 images:")
for file in image_files[:5]:
    print(file.name)

print("\nFirst 5 masks:")
for file in mask_files[:5]:
    print(file.name)