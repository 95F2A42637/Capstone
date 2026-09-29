import os
import shutil
import random

SOURCE = {
    "authentic": r"dataset\authentic",
    "ai_edited": r"dataset\ai_edited"
}

DEST = {
    "train": {
        "authentic": r"dataset\train\authentic",
        "ai_edited": r"dataset\train\ai_edited"
    },
    "val": {
        "authentic": r"dataset\val\authentic",
        "ai_edited": r"dataset\val\ai_edited"
    },
    "test": {
        "authentic": r"dataset\test\authentic",
        "ai_edited": r"dataset\test\ai_edited"
    }
}

# Actual project training subset
# 10,000 images per class = 20,000 total
TRAIN_PER_CLASS = 8000
VAL_PER_CLASS = 1000
TEST_PER_CLASS = 1000

random.seed(42)

for class_name in SOURCE:
    files = [
        f for f in os.listdir(SOURCE[class_name])
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    random.shuffle(files)

    required = TRAIN_PER_CLASS + VAL_PER_CLASS + TEST_PER_CLASS

    if len(files) < required:
        raise ValueError(
            f"{class_name} has only {len(files)} images, "
            f"but {required} are required."
        )

    selected = files[:required]

    splits = {
        "train": selected[:TRAIN_PER_CLASS],
        "val": selected[TRAIN_PER_CLASS:TRAIN_PER_CLASS + VAL_PER_CLASS],
        "test": selected[TRAIN_PER_CLASS + VAL_PER_CLASS:]
    }

    for split_name, split_files in splits.items():
        destination = DEST[split_name][class_name]

        for filename in split_files:
            source_path = os.path.join(SOURCE[class_name], filename)
            destination_path = os.path.join(destination, filename)

            if not os.path.exists(destination_path):
                shutil.copy2(source_path, destination_path)

        print(f"{class_name} -> {split_name}: {len(split_files)} images")

print("\nDataset preparation completed successfully.")
print("Total selected images: 20,000")