# YOLOv8 detection module
# We will use this file for vehicle detection and later model validation/training.

from ultralytics import YOLO

def load_model(weights="models/yolov8n.pt"):
    return YOLO(weights)

if __name__ == "__main__":
    model = load_model()
    print("YOLOv8 model loaded successfully.")
