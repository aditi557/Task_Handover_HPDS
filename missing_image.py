import os


IMAGE_DIR = "/val/images"
LABEL_DIR = "/val/labels"

# Define your extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
LABEL_EXTENSION = ".txt"

def find_missing_images():
    # 1. Get all image base names (filename without extension)
    image_basenames = set()
    for file in os.listdir(IMAGE_DIR):
        name, ext = os.path.splitext(file)
        if ext.lower() in IMAGE_EXTENSIONS:
            image_basenames.add(name)

    # 2. Find labels that don't have a matching image
    missing_images = []
    for file in os.listdir(LABEL_DIR):
        name, ext = os.path.splitext(file)
        
        if ext.lower() == LABEL_EXTENSION.lower():
            if name not in image_basenames:
                missing_images.append(name)

    # 3. Print the report
    print(f"Total valid images found: {len(image_basenames)}")
    print(f"Total annotations checked: {len(os.listdir(LABEL_DIR))}")
    print("-" * 40)
    
    if missing_images:
        print(f"Found {len(missing_images)} annotations missing their images:")
        for img_name in sorted(missing_images):
            # Prints the name (e.g., 'frame_001')
            print(f" Missing image for: {img_name}{LABEL_EXTENSION}") 
    else:
        print("Success! All annotation files have a corresponding image.")

if __name__ == "__main__":
    find_missing_images()
