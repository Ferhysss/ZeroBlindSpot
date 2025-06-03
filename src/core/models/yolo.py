from ultralytics import YOLO
import logging

class YoloModel:
    def __init__(self, model_path: str):
        try:
            self.model = YOLO(model_path)
            logging.info(f"YOLO model loaded: {model_path}")
        except Exception as e:
            logging.error(f"Failed to load YOLO: {str(e)}")
            raise

    def predict(self, frame):
        return self.model(frame)

    def track(self, frame, persist: bool = True):
        return self.model.track(frame, persist=persist)