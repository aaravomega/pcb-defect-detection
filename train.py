from roboflow import Roboflow
from ultralytics import YOLO

# Download dataset from Roboflow
rf = Roboflow(api_key="YOUR_ROBOFLOW_API_KEY")
project = rf.workspace("tcc-r4j2r").project("pcb-la0tj")
version = project.version(2)
dataset = version.download("yolov8")

# Load pretrained YOLOv8 nano model
model = YOLO("yolov8n.pt")

# Train
model.train(
    data=f"{dataset.location}/data.yaml",
    epochs=50,
    imgsz=608,
    batch=16,
    name="pcb_defect_detector"
)

# Evaluate on test set
metrics = model.val(split="test")
print(f"mAP50: {metrics.box.map50:.3f}")
print(f"mAP50-95: {metrics.box.map:.3f}")
