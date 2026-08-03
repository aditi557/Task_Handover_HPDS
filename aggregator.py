#aggregates images from multiple folders ino one big folder

from pathlib import Path
import shutil

source_dir = Path("Source_Folder")

# Destination folder
destination_dir = Path("Destination_Directory")

image_extensions = {".jpg", ".jpeg", ".png"}

count = 0

for file_path in source_dir.rglob("*"):
    if '0' in file_path.parts:
        continue
    
    if file_path.suffix.lower() in image_extensions:
        
        # Handle duplicate filenames
        destination_file = destination_dir / file_path.name
        
        if destination_file.exists():
            stem = file_path.stem
            suffix = file_path.suffix
            i = 1
            
            while destination_file.exists():
                destination_file = destination_dir / f"{stem}_{i}{suffix}"
                i += 1

        # Copy file
        shutil.copy2(file_path, destination_file)

        # If you want to MOVE instead of copy:
        # shutil.move(str(file_path), str(destination_file))

        count += 1

print(f"{count} images combined into {destination_dir}")