import os
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from ultralytics import YOLO

# =========================
# CONFIG
# =========================
DATA_YAML = "data.yaml"
RUNS_DIR = "runs_train"
EXP_NAME = "exp"
LABEL_DIR = "/train/labels"


# =========================
# 1. TRAIN MODEL
# =========================
def train_model():
    print("TRAINING BEGINS")

    model = YOLO("Path to Model")

    model.train(
        data=DATA_YAML,
        imgsz=640,
        epochs=50,
        patience=10,
        batch=8,
        workers=0,
        project=RUNS_DIR,
        name=EXP_NAME,
        exist_ok=True
    )

    print("TRAINING FINISHED")


# =========================
# 2. DATASET ANALYSIS
# =========================
def analyze_dataset(label_dir):
    print("\n📊 DATASET ANALYSIS")

    class_counts = defaultdict(int)
    total_objects = 0
    total_images = 0

    for file in os.listdir(label_dir):
        if file.endswith(".txt"):
            total_images += 1
            with open(os.path.join(label_dir, file), "r") as f:
                for line in f:
                    cls = int(line.split()[0])
                    class_counts[cls] += 1
                    total_objects += 1

    print(f"Total Images: {total_images}")
    print(f"Total Objects: {total_objects}")
    print(f"Avg objects/image: {total_objects / total_images:.2f}")

    print("\nClass Distribution:")
    for k, v in class_counts.items():
        print(f"Class {k}: {v}")

    return class_counts, total_images, total_objects


# =========================
# 3. PLOT CLASS DISTRIBUTION
# =========================
def plot_class_distribution(class_counts):
    plt.figure()
    classes = list(class_counts.keys())
    counts = list(class_counts.values())

    plt.bar(classes, counts)
    plt.xlabel("Class")
    plt.ylabel("Frequency")
    plt.title("Class Distribution")

    save_path = os.path.join(RUNS_DIR, EXP_NAME, "class_distribution.png")
    plt.savefig(save_path)
    plt.close()

    print(f"📈 Saved class distribution → {save_path}")


# =========================
# 4. LEARNING CURVES
# =========================
def plot_learning_curves():
    print("\n📉 GENERATING LEARNING CURVES")

    results_path = "/runs/detect/runs_train/exp/results.csv"

    df = pd.read_csv(results_path)

    # ---- mAP curves ----
    plt.figure()
    plt.plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@0.5')
    plt.plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP@0.5:0.95')
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend()
    plt.title("Validation Performance")


    # ---- Loss curves ----
    plt.figure()
    plt.plot(df['epoch'], df['train/box_loss'], label='Box Loss')
    plt.plot(df['epoch'], df['train/cls_loss'], label='Cls Loss')
    plt.plot(df['epoch'], df['train/dfl_loss'], label='DFL Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training Loss")

    # loss_path = os.path.join(RUNS_DIR, EXP_NAME, "loss_curve.png")
    # plt.savefig(loss_path)
    # plt.close()

    # print(f"📈 Saved mAP curve → {map_path}")
    # print(f"📉 Saved loss curve → {loss_path}")


# =========================
# 5. SUMMARY REPORT
# =========================
def generate_report(class_counts, total_images, total_objects):
    print("\n🧾 GENERATING REPORT")

    results_path = os.path.join(RUNS_DIR, EXP_NAME, "results.csv")
    df = pd.read_csv(results_path)

    best_epoch = df['metrics/mAP50(B)'].idxmax()
    best_map = df.loc[best_epoch, 'metrics/mAP50(B)']

    report = f"""
================ YOLOv8 TRAINING REPORT ================

Dataset:
- Total Images: {total_images}
- Total Objects: {total_objects}
- Avg Objects/Image: {total_objects / total_images:.2f}

Class Distribution:
{dict(class_counts)}

Training:
- Total Epochs: {len(df)}
- Early Stopping Patience: 30
- Best Epoch: {best_epoch}
- Best mAP@0.5: {best_map:.4f}

=======================================================
"""

    report_path = os.path.join(RUNS_DIR, EXP_NAME, "report_engine_25_05_26).txt")
    with open(report_path, "w") as f:
        f.write(report)

    print(report)
    print(f"📄 Report saved → {report_path}")


# =========================
# MAIN PIPELINE
# =========================
if __name__ == "__main__":

    # 1. Train
    train_model()
    plot_learning_curves()
    # # 2. Dataset stats
    # class_counts, total_images, total_objects = analyze_dataset("labels")

    # # 3. Visualizations
    # plot_class_distribution(class_counts)


    # # 4. Report
    # generate_report(class_counts, total_images, total_objects)