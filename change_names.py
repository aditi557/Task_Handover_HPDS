import os
import random
import shutil
from datetime import datetime

def rename_and_split_images(source_dir, output_dir):
    # Reproducibility (optional)
    random.seed(42)

    # Get today's date
    date_str = datetime.now().strftime("%d_%m_%Y")

    # Create output folders
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    test_dir = os.path.join(output_dir, "test")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Collect image files
    files = [f for f in os.listdir(source_dir)
             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # Shuffle randomly
    random.shuffle(files)

    total = len(files)
    train_split = int(0.8 * total)
    val_split = int(0.9 * total)

    count = 1

    for i, filename in enumerate(files):
        old_path = os.path.join(source_dir, filename)

        # New renamed file
        new_name = f"{date_str}_{count}.jpg"

        # Decide destination
        if i < train_split:
            target_dir = train_dir
        elif i < val_split:
            target_dir = val_dir
        else:
            target_dir = test_dir

        new_path = os.path.join(target_dir, new_name)

        # Move + rename
        shutil.move(old_path, new_path)

        print(f"{filename} -> {new_name} ({os.path.basename(target_dir)})")

        count += 1

    print("\nSummary:")
    print(f"Total: {total}")
    print(f"Train: {train_split}")
    print(f"Val: {val_split - train_split}")
    print(f"Test: {total - val_split}")


#Sample Usage

source_folder = "source_file_path"
output_folder = "output_file_path"

rename_and_split_images(source_folder, output_folder)