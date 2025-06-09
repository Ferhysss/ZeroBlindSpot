from PyQt5.QtCore import QThread, pyqtSignal
import cv2
import os
import logging
from typing import List, Tuple, Optional
from core.models.yolo_model import YoloModel
from core.models.cnn import SimpleCNN
from core.config import Config
import torch
from ultralytics import YOLO
from PyQt5.QtCore import QObject, pyqtSignal
import cv2
import torch
import numpy as np
import os
import logging
from typing import List, Tuple, Optional

class DeveloperProcessor(QObject):
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    frame_processed = pyqtSignal(int, np.ndarray)
    no_bucket_frame = pyqtSignal(int, np.ndarray)
    low_conf_frame = pyqtSignal(int, np.ndarray, list)  # Добавлен третий аргумент (list)
    finished = pyqtSignal()

    def __init__(self, video_path: str, yolo_model, cnn_model, config, project_dir: str):
        super().__init__()
        self.video_path = video_path
        self.yolo_model = yolo_model
        self.cnn_model = cnn_model
        self.config = config
        self.project_dir = project_dir

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.status.emit("Error: failed to open video")
            logging.error(f"Failed to open video: {self.video_path}")
            return

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_rate = self.config.get("frame_rate", 1)
        cycles = []
        current_cycle = None

        for frame_idx in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_rate != 0:
                continue

            results = self.yolo_model.predict(frame)
            bucket_detected = False

            for box in results[0]:  # results[0] — список кортежей
                x, y, w, h, conf, class_id = box
                if conf < self.config.get("yolo_conf_threshold", 0.5):
                    self.low_conf_frame.emit(frame_idx, frame, [(x, y, w, h, conf, class_id)])
                    logging.info(f"Low confidence frame: {frame_idx}, conf: {conf}")
                    continue
                bucket_detected = True
                frame_path = os.path.join(self.project_dir, "frames", f"{frame_idx:06d}.jpg")
                cv2.imwrite(frame_path, frame)

                if self.cnn_model:
                    crop = frame[int(y - h / 2):int(y + h / 2), int(x - w / 2):int(x + w / 2)]
                    if crop.size == 0:
                        continue
                    crop = cv2.resize(crop, (224, 224)) / 255.0
                    crop = crop.transpose((2, 0, 1))
                    crop_tensor = torch.tensor(crop, dtype=torch.float32).unsqueeze(0).to(self.cnn_model.device)
                    with torch.no_grad():
                        output = self.cnn_model(crop_tensor)
                        class_id = output.argmax(dim=1).item()
                        logging.info(f"CNN predicted class: {class_id} for frame {frame_idx}")
                    if class_id == 1:  # Зачерпывание
                        if current_cycle is None:
                            current_cycle = {"start": frame_idx, "excavator": self.config.get("excavator", "Excavator A")}
                    elif class_id == 2 and current_cycle:  # Высыпание
                        current_cycle["end"] = frame_idx
                        cycles.append(current_cycle)
                        current_cycle = None
                        self.status.emit(f"Cycle detected: {len(cycles)}")
                break

            if not bucket_detected:
                self.no_bucket_frame.emit(frame_idx, frame)
                logging.info(f"No bucket detected in frame {frame_idx}")

            progress = int((frame_idx + 1) / frame_count * 100)
            self.progress.emit(progress)

        cap.release()
        self._save_results(cycles)
        self.finished.emit()

    def _save_results(self, cycles: List[dict]):
        os.makedirs(os.path.join(self.project_dir, "results"), exist_ok=True)
        bucket_volume = self.config.get("bucket_volume", 1.5)
        total_volume = len(cycles) * bucket_volume
        with open(os.path.join(self.project_dir, "results", "summary.txt"), "w", encoding='utf-8') as f:
            f.write(f"Excavator: {self.config.get('excavator', 'Excavator A')}\n")
            f.write(f"Bucket volume: {bucket_volume} m³\n")
            f.write(f"Total cycles: {len(cycles)}\n")
            f.write(f"Total volume: {total_volume} m³\n")
        logging.info(f"Results saved: {len(cycles)} cycles, {total_volume} m³")

    def extract_frames(self, video_path: str):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.status.emit("Ошибка: не удалось открыть видео")
            return
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0
        os.makedirs(os.path.join(self.project_dir, "frames"), exist_ok=True)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_path = os.path.join(self.project_dir, "frames", f"frame_{frame_count:04d}.jpg")
            cv2.imwrite(frame_path, frame)
            self.progress.emit(int((frame_count / total_frames) * 100))
            frame_count += 1
        cap.release()
        self.status.emit("Извлечение кадров завершено")

    def stop(self):
        self.is_running = False

class YoloPredictor(QObject):
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    frame_processed = pyqtSignal(int, np.ndarray)
    no_bucket_frame = pyqtSignal(int, np.ndarray)
    low_conf_frame = pyqtSignal(int, np.ndarray, list)  # Подтверждаем сигнатуру
    finished = pyqtSignal()