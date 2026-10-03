import torch


def dice_loss(logits, targets, smooth=1e-6):

    # Convert logits to probabilities
    probabilities = torch.sigmoid(logits)

    # Flatten
    probabilities = probabilities.view(-1)
    targets = targets.view(-1)

    # Intersection
    intersection = (probabilities * targets).sum()

    # Dice coefficient
    dice = (
        (2.0 * intersection + smooth)
        /
        (
            probabilities.sum()
            + targets.sum()
            + smooth
        )
    )

    return 1.0 - dice


def combined_loss(logits, targets):

    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        targets
    )

    dice = dice_loss(
        logits,
        targets
    )

    return bce + dice