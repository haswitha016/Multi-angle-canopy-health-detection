"""Dataset validation and RGB image transformations."""

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from inference.fusion import CLASS_NAMES


IMAGE_SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def image_transforms(training: bool = False) -> transforms.Compose:
    operations = []
    if training:
        operations.extend(
            [
                transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.75, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.15, hue=0.03),
            ]
        )
    else:
        operations.extend([transforms.Resize(256), transforms.CenterCrop(IMAGE_SIZE)])
    operations.extend([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    return transforms.Compose(operations)


def build_loaders(dataset_dir: str | Path, batch_size: int = 16, num_workers: int = 0):
    """Load train/validation/test folders and verify their class mappings."""
    root = Path(dataset_dir)
    loaders = {}
    expected = list(CLASS_NAMES)
    for split, training in (("train", True), ("validation", False), ("test", False)):
        split_path = root / split
        if not split_path.is_dir():
            raise FileNotFoundError(f"Missing dataset split: {split_path}")
        dataset = datasets.ImageFolder(split_path, transform=image_transforms(training))
        if dataset.classes != expected:
            raise ValueError(
                f"{split_path} classes must be exactly {expected}, found {dataset.classes}"
            )
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=training,
            num_workers=num_workers,
            pin_memory=False,
        )
    return loaders