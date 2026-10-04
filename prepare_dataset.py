import io
import random
from pathlib import Path

import pandas as pd
from PIL import Image


PARQUET_FILE = Path.home() / "Downloads" / "canopy_dataset.parquet"
DATA_DIR = Path("data")

import io
import random
from pathlib import Path

import pandas as pd
from PIL import Image


PARQUET_FILE = Path.home() / "Downloads" / "canopy_dataset.parquet"
DATA_DIR = Path("data")

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15

random.seed(42)


print("Loading dataset...")
df = pd.read_parquet(PARQUET_FILE)

print(f"Total images: {len(df)}")
print(df["label"].value_counts())


# Shuffle dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Split dataset
train_end = int(len(df) * TRAIN_RATIO)
validation_end = int(len(df) * (TRAIN_RATIO + VALIDATION_RATIO))

train_df = df[:train_end]
validation_df = df[train_end:validation_end]
test_df = df[validation_end:]

splits = {
    "train": train_df,
    "validation": validation_df,
    "test": test_df,
}


for split_name, split_df in splits.items():

    print(f"\nPreparing {split_name} set...")
    print(f"Images: {len(split_df)}")

    for index, row in split_df.iterrows():

        if row["label"] == "healthy":
            class_name = "Healthy"
        else:
            class_name = "Stressed"

        output_dir = DATA_DIR / split_name / class_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Extract image bytes from the dictionary
        image_data = row["image"]

        if isinstance(image_data, dict):
            image_data = image_data["bytes"]

        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        output_file = output_dir / f"{split_name}_{index}.jpg"

        image.save(output_file, "JPEG")

print("\nDataset preparation complete!")

