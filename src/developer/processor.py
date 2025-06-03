import os
import cv2
import torch
import numpy as np
import ffmpeg
from PyQt5.QtCore import QThread, pyqtSignal
from core.config import Config
from core.models.yolo import YoloModel
from core.models.cnn import SimpleCNN
from typing import Optional, List, Tuple
import logging
from datetime import datetime

class DeveloperProcessor(QThread):
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished = pyqtSignal()
    frame_processed = pyqtSignal(str, list)
    no_bucket_frame = pyqtSignal(str)
    low_conf_frame = pyqtSignal(str, list)

    def __init__(self, video_path: str, yolo_model: YoloModel, cnn_model: Optional[SimpleCNN], config: Config, output_dir: str):
        super().__init__()
        self.video_path = video_path
        self.yolo_model = yolo_model
        self.cnn_model = cnn_model
        self.config = config
        self.output_dir = output_dir

    def run(self) -> None:
        try:
            self.status.emit("Разбиение видео...")
            frame_dir = f"{self.output_dir}/frames"
            no_bucket_dir = f"{self.output_dir}/no_bucket"
            os.makedirs(frame_dir, exist_ok=True)
            os.makedirs(no_bucket_dir, exist_ok=True)
            frame_rate = self.config.get("frame_rate", 1)
            stream = ffmpeg.input(self.video_path)
            stream = ffmpeg.output(stream, f"{frame_dir}/frame_%04d.jpg", r=frame_rate)
            ffmpeg.run(stream, overwrite_output=True)
            logging.info(f"Frames saved in {frame_dir}")

            frame_files = sorted(os.listdir(frame_dir))
            total_frames = len(frame_files)
            for i, frame_file in enumerate(frame_files):
                frame_path = os.path.join(frame_dir, frame_file)
                frame = cv2.imread(frame_path)
                if frame is None:
                    continue

                results = self.yolo_model.predict(frame)
                boxes = results[0].boxes.xywh.cpu().numpy()
                confidences = results[0].boxes.conf.cpu().numpy()
                frame_annotations: List[Tuple[float, float, float, float, float]] = []
                low_conf = False
                for box, conf in zip(boxes, confidences):
                    if conf < self.config.get("yolo_conf_threshold", 0.5):
                        continue
                    x, y, w, h = box
                    frame_annotations.append((x, y, w, h, conf))
                    if conf < 0.6:
                        low_conf = True

                if not frame_annotations:
                    no_bucket_path = os.path.join(no_bucket_dir, frame_file)
                    os.rename(frame_path, no_bucket_path)
                    self.no_bucket_frame.emit(no_bucket_path)
                    continue
                if low_conf:
                    self.low_conf_frame.emit(frame_path, frame_annotations)

                self.frame_processed.emit(frame_path, frame_annotations)
                self.progress.emit(int((i + 1) / total_frames * 100))

            self.status.emit("Обработка завершена!")
            logging.info("Processing completed")
        except Exception as e:
            self.status.emit(f"Ошибка: {str(e)}")
            logging.error(f"Processing failed: {str(e)}")
        finally:
            self.finished.emit()

    def extract_frames(self, video_path: str) -> None:
        """Извлекает кадры из видео для аннотации."""
        try:
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            output_dir = f"data/frames_{timestamp}"
            os.makedirs(output_dir, exist_ok=True)
            cap = cv2.VideoCapture(video_path)
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                cv2.imwrite(f"{output_dir}/frame_{frame_count:04d}.jpg", frame)
                frame_count += 1
            cap.release()
            self.status.emit(f"Извлечено {frame_count} кадров в {output_dir}")
            logging.info(f"Extracted {frame_count} frames to {output_dir}")
        except Exception as e:
            self.status.emit(f"Ошибка извлечения: {str(e)}")
            logging.error(f"Frame extraction failed: {str(e)}")