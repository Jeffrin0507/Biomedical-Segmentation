from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as TF


# ============================================================
# 1. PATHS
# ============================================================

DATASET_DIR = Path(
    "dataset/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG"
)

IMAGE_DIR = DATASET_DIR / "images"
MASK_DIR = DATASET_DIR / "masks"

SPLIT_DIR = Path("dataset/Kvasir-SEG")

TRAIN_FILE = SPLIT_DIR / "train.txt"
VAL_FILE = SPLIT_DIR / "val.txt"


# ============================================================
# 2. SETTINGS
# ============================================================

IMAGE_SIZE = (256, 256)

RANDOM_SEED = 42

BATCH_SIZE = 8


# ============================================================
# 3. READ DATASET SPLITS
# ============================================================

def read_split_file(file_path):
    """
    Read image IDs from train.txt or val.txt.
    """

    with open(file_path, "r") as f:
        ids = [line.strip() for line in f if line.strip()]

    return ids


# ============================================================
# 4. CREATE TRAIN / VALIDATION / TEST SPLIT
# ============================================================

def create_splits():

    train_ids = read_split_file(TRAIN_FILE)
    val_ids = read_split_file(VAL_FILE)

    # The dataset provides:
    # 880 training images
    # 120 validation images
    #
    # We split the original 880 training images into:
    # 704 training
    # 176 testing

    random.seed(RANDOM_SEED)

    random.shuffle(train_ids)

    test_ids = train_ids[:176]
    final_train_ids = train_ids[176:]

    print("Dataset split:")
    print(f"Training   : {len(final_train_ids)}")
    print(f"Validation : {len(val_ids)}")
    print(f"Testing    : {len(test_ids)}")

    return final_train_ids, val_ids, test_ids


# ============================================================
# 5. DATASET CLASS
# ============================================================

class KvasirDataset(Dataset):

    def __init__(
        self,
        image_ids,
        image_size=IMAGE_SIZE,
        augment=False
    ):

        self.image_ids = image_ids
        self.image_size = image_size
        self.augment = augment

    def __len__(self):

        return len(self.image_ids)

    def __getitem__(self, index):

        image_id = self.image_ids[index]

        # ----------------------------------------------------
        # Image and mask paths
        # ----------------------------------------------------

        image_path = IMAGE_DIR / f"{image_id}.jpg"
        mask_path = MASK_DIR / f"{image_id}.jpg"

        # ----------------------------------------------------
        # Load image
        # ----------------------------------------------------

        image = Image.open(image_path).convert("RGB")

        # ----------------------------------------------------
        # Load mask
        # ----------------------------------------------------

        mask = Image.open(mask_path).convert("L")

        # ----------------------------------------------------
        # Resize
        # ----------------------------------------------------

        image = TF.resize(
            image,
            self.image_size,
            interpolation=TF.InterpolationMode.BILINEAR
        )

        mask = TF.resize(
            mask,
            self.image_size,
            interpolation=TF.InterpolationMode.NEAREST
        )

        # ----------------------------------------------------
        # Data augmentation
        # ----------------------------------------------------

        if self.augment:

            # Random horizontal flip
            if random.random() > 0.5:

                image = TF.hflip(image)
                mask = TF.hflip(mask)

            # Random vertical flip
            if random.random() > 0.5:

                image = TF.vflip(image)
                mask = TF.vflip(mask)

        # ----------------------------------------------------
        # Convert image to tensor
        # ----------------------------------------------------

        image = TF.to_tensor(image)

        # ----------------------------------------------------
        # Normalize image
        #
        # Image values:
        # 0-255 → 0-1
        # ----------------------------------------------------

        image = TF.normalize(
            image,
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )

        # ----------------------------------------------------
        # Convert mask to tensor
        # ----------------------------------------------------

        mask = np.array(mask)

        # Binary segmentation
        #
        # Background → 0
        # Polyp      → 1
        #

        mask = (mask >= 128).astype(np.float32)

        mask = torch.from_numpy(mask)

        # Add channel dimension
        #
        # H × W
        #   ↓
        # 1 × H × W

        mask = mask.unsqueeze(0)

        return image, mask


# ============================================================
# 6. CREATE DATASETS AND DATALOADERS
# ============================================================

if __name__ == "__main__":

    train_ids, val_ids, test_ids = create_splits()

    train_dataset = KvasirDataset(
        train_ids,
        augment=True
    )

    val_dataset = KvasirDataset(
        val_ids,
        augment=False
    )

    test_dataset = KvasirDataset(
        test_ids,
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

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # --------------------------------------------------------
    # Test one batch
    # --------------------------------------------------------

    images, masks = next(iter(train_loader))

    print("\nBatch information:")
    print("Image shape:", images.shape)
    print("Mask shape :", masks.shape)

    print("\nImage dtype:", images.dtype)
    print("Mask dtype :", masks.dtype)

    print("\nMask unique values:", torch.unique(masks))