from PyQt5.QtCore import QThread, pyqtSignal
import cv2
import os
import logging
from typing import List, Tuple, Optional
from core.models.yolo import YoloModel
from core.models.cnn import SimpleCNN
from core.config import Config
import torch
from ultralytics import YOLO

class DeveloperProcessor(QThread):
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    frame_processed = pyqtSignal(str, list)
    no_bucket_frame = pyqtSignal(str)
    low_conf_frame = pyqtSignal(str, list)
    finished = pyqtSignal()

    def __init__(self, video_path: str, yolo_model: YoloModel, cnn_model: Optional[SimpleCNN], config: Config, project_dir: str):
        super().__init__()
        self.video_path = video_path
        self.yolo_model = yolo_model
        self.cnn_model = cnn_model
        self.config = config
        self.project_dir = project_dir
        self.is_running = True

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.status.emit("Ошибка: не удалось открыть видео")
            return
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_rate = self.config.get("frame_rate", 1)
        frame_count = 0
        os.makedirs(os.path.join(self.project_dir, "frames"), exist_ok=True)
        os.makedirs(os.path.join(self.project_dir, "no_bucket"), exist_ok=True)
        os.makedirs(os.path.join(self.project_dir, "annotations"), exist_ok=True)
        os.makedirs(os.path.join(self.project_dir, "cnn_annotations"), exist_ok=True)
        os.makedirs(os.path.join(self.project_dir, "results"), exist_ok=True)

        cycles = 0
        prev_state = None

        while self.is_running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_rate == 0:
                frame_path = os.path.join(self.project_dir, "frames", f"frame_{frame_count:04d}.jpg")
                cv2.imwrite(frame_path, frame)
                results = self.yolo_model.predict(frame)  # YOLOv8 predict
                annotations = []
                for box in results[0].boxes:  # results[0] — первый результат батча
                    x, y, w, h = box.xywh[0].cpu().numpy()  # Центр, ширина, высота
                    conf = box.conf[0].cpu().numpy()  # Уверенность
                    class_id = int(box.cls[0].cpu().numpy())  # ID класса
                    annotations.append((x, y, w, h, conf, class_id))
                self.frame_processed.emit(frame_path, annotations)
                if not annotations:
                    no_bucket_path = os.path.join(self.project_dir, "no_bucket", f"frame_{frame_count:04d}.jpg")
                    cv2.imwrite(no_bucket_path, frame)
                    self.no_bucket_frame.emit(no_bucket_path)
                elif any(conf < 0.5 for _, _, _, _, conf, _ in annotations):
                    self.low_conf_frame.emit(frame_path, annotations)
                if self.cnn_model and annotations:
                    cnn_annotation_path = os.path.join(self.project_dir, "cnn_annotations", f"frame_{frame_count:04d}.jpg.txt")
                    for x, y, w, h, _, _ in annotations:
                        x1 = int(x - w / 2)
                        y1 = int(y - h / 2)
                        x2 = int(x + w / 2)
                        y2 = int(y + h / 2)
                        region = frame[y1:y2, x1:x2]
                        if region.size == 0:
                            continue
                        region = cv2.resize(region, (224, 224))
                        region = region / 255.0
                        region = region.transpose((2, 0, 1))
                        region_tensor = torch.tensor(region, dtype=torch.float32).unsqueeze(0).to(self.cnn_model.device)
                        with torch.no_grad():
                            output = self.cnn_model(region_tensor)
                            class_id = torch.argmax(output, dim=1).item()
                        with open(cnn_annotation_path, "a", encoding='utf-8') as f:
                            f.write(f"{class_id}\n")
                        logging.info(f"CNN predicted class {class_id} for {frame_path}")
                        # Подсчёт циклов
                        if class_id == 2 and prev_state == 1:  # Высыпание после Зачерпывания
                            cycles += 1
                        prev_state = class_id
                self.progress.emit(int((frame_count / total_frames) * 100))
            frame_count += 1
        cap.release()
        excavator = self.config.get("excavator", "Экскаватор A")
        bucket_volume = self.config.get("bucket_volume", 1.5)
        total_volume = cycles * bucket_volume
        result_path = os.path.join(self.project_dir, "results", "summary.txt")
        with open(result_path, "w", encoding='utf-8') as f:
            f.write(f"Экскаватор: {excavator}\n")
            f.write(f"Объём ковша: {bucket_volume} м³\n")
            f.write(f"Циклов (Зачерпывание → Высыпание): {cycles}\n")
            f.write(f"Общий объём грунта: {total_volume} м³\n")
        logging.info(f"Results saved to {result_path}")
        self.status.emit(f"Обработка завершена: {cycles} циклов, {total_volume} м³")
        self.finished.emit()

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