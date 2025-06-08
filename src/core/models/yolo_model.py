from ultralytics import YOLO
import torch

class YoloModel:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def predict(self, frame):
        results = self.model(frame, verbose=False)
        boxes = []
        for box in results[0].boxes:
            x, y, w, h = box.xywh[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            class_id = int(box.cls[0].cpu().numpy())
            boxes.append((x, y, w, h, conf, class_id))
        return [boxes]