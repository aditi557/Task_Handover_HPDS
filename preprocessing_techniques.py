import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def apply_clahe(img, tileGridSize = (4, 4)):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    y_clahe = clahe.apply(y)

    # clahe2 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    # y_clahe2 = clahe2.apply(y)

    merged = cv2.merge((y_clahe, cr, cb))
    return cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)

def histogram_equalization(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eq = cv2.equalizeHist(gray)
    return cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)

def gamma_correction(img, gamma=1.5):
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(img, table)

def gaussian_blur(img):
    return cv2.GaussianBlur(img, (5, 5), 0)

def sharpening(img):
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)

def process_image(img_path, output_dir):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error loading {img_path}")
        return

    img_name = os.path.basename(img_path).split('.')[0]

    techniques = {
        "original": img,
        "clahe": apply_clahe(img),
        "hist_eq": histogram_equalization(img),
        "gamma": gamma_correction(img, gamma=1.5),
        "blur": gaussian_blur(img),
        "sharpen": sharpening(img)
    }

    os.makedirs(output_dir, exist_ok=True)
    for name, processed in techniques.items():
        save_path = os.path.join(output_dir, f"{img_name}_{name}.jpg")
        cv2.imwrite(save_path, processed)


    plt.figure(figsize=(12, 8))
    for i, (name, processed) in enumerate(techniques.items()):
        plt.subplot(2, 3, i+1)
        plt.imshow(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
        plt.title(name)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def process_folder(input_folder, output_folder):
    for file in os.listdir(input_folder):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            process_image(
                os.path.join(input_folder, file),
                output_folder
            )

input_folder = "output_night_images"
output_folder = "output_preprocessed_images"

process_folder(input_folder, output_folder)
