from ultralytics import YOLO
from core.config import Config
import logging
from typing import Optional

class Trainer:
    def __init__(self, config: Config):
        self.config = config
        self.model: Optional[YOLO] = None

    def train_yolo(self, data_path: str, epochs: int = 10) -> None:
        """Обучает модель YOLO на аннотированных данных."""
        try:
            self.model = YOLO("models/model.pt")
            self.model.train(
                data=data_path,
                epochs=epochs,
                imgsz=640,
                device=self.config.get("device", "auto"),
                batch=16,
                name=f"train_{self.config.get('project_name', 'default')}"
            )
            logging.info(f"YOLO training completed: {data_path}")
        except Exception as e:
            logging.error(f"YOLO training failed: {str(e)}")
            raise