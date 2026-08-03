#This script is for inferencing over a directory of images using a finetuned pytorch model

from ultralytics import YOLO

model_path = "/runs/detect/runs_train/exp/weights/best.pt"
data_yaml = "data.yaml"



model = YOLO(model_path)

# -------- METRICS -------- #

print("\n===== METRICS =====")

day_metrics = model.val(
    data=data_yaml,
    split="test",
    conf=0.5, #set as per requirement
    save=True,
    name="Type_test"
)


print(f"mAP50: {day_metrics.box.map50:.4f}")
print(f"Precision: {day_metrics.box.mp:.4f}")
print(f"Recall: {day_metrics.box.mr:.4f}")

p, r = day_metrics.box.mp, day_metrics.box.mr
f1 = 2 * (p * r) / (p + r + 1e-6)
print(f"F1: {f1:.4f}")