import os
import cv2
import torch
import numpy as np
import ffmpeg
import sqlite3
from datetime import datetime
from PyQt5.QtCore import QThread, pyqtSignal
from core.config import Config
from core.models.yolo import YoloModel
from core.models.cnn import SimpleCNN
from typing import Optional, List, Tuple
import logging

class OperatorProcessor(QThread):
    """Обработчик видео для модуля оператора: детекция ковша, подсчёт циклов, расчёт выработки."""
    
    progress = pyqtSignal(int)
    status_signal = pyqtSignal(str)
    finished = pyqtSignal()
    cycle_count = pyqtSignal(int, float)  # Сигнал для циклов и выработки (циклы, м³)

    def __init__(self, video_path: str, yolo_model: YoloModel, cnn_model: Optional[SimpleCNN], config: Config, output_dir: str):
        """
        Инициализация процессора.

        Args:
            video_path: Путь к видеофайлу.
            yolo_model: Модель YOLO для детекции ковша.
            cnn_model: Модель CNN для классификации состояний (опционально).
            config: Конфигурация приложения.
            output_dir: Директория для сохранения результатов.
        """
        super().__init__()
        self.video_path = video_path
        self.yolo_model = yolo_model
        self.cnn_model = cnn_model
        self.config = config
        self.output_dir = output_dir
        self.cycles: int = 0
        self.last_state: Optional[str] = None
        self.bucket_volume: float = self.config.get("bucket_volume", 1.0)

    def run(self) -> None:
        """Основной метод обработки видео."""
        try:
            # Разбиение видео на кадры
            self.status_signal.emit("Разбиение видео...")
            frame_dir = f"{self.output_dir}/frames"
            os.makedirs(frame_dir, exist_ok=True)
            frame_rate = self.config.get("frame_rate", 1)
            stream = ffmpeg.input(self.video_path)
            stream = ffmpeg.output(stream, f"{frame_dir}/frame_%04d.jpg", r=str(frame_rate))
            ffmpeg.run(stream, overwrite_output=True)
            logging.info(f"Frames saved in {frame_dir}")

            # Инициализация SQLite для логов
            conn = sqlite3.connect(f"{self.output_dir}/logs.db")
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS events 
                        (frame TEXT, class TEXT, conf REAL, track_id INTEGER, timestamp TEXT)''')
            conn.commit()

            # Обработка кадров
            frame_files = sorted(os.listdir(frame_dir))
            total_frames = len(frame_files)
            for i, frame_file in enumerate(frame_files):
                frame_path = os.path.join(frame_dir, frame_file)
                frame = cv2.imread(frame_path)
                if frame is None:
                    logging.warning(f"Failed to read frame: {frame_path}")
                    continue

                # YOLO с трекингом
                results = self.yolo_model.track(frame, persist=True)
                boxes = results[0].boxes.xywh.cpu().numpy()
                confidences = results[0].boxes.conf.cpu().numpy()
                track_ids = (
                    results[0].boxes.id.cpu().numpy()
                    if results[0].boxes.id is not None
                    else np.array([], dtype=np.int64)
                )

                frame_annotations: List[Tuple[float, float, float, float, float]] = []
                for box, conf, track_id in zip(boxes, confidences, track_ids):
                    if conf < self.config.get("yolo_conf_threshold", 0.5):
                        continue
                    x, y, w, h = box
                    frame_annotations.append((x, y, w, h, conf))

                    # CNN классификация
                    state_label = "neutral"
                    if self.config.get("use_cnn", False) and self.cnn_model:
                        x1, y1 = int(x - w / 2), int(y - h / 2)
                        x2, y2 = int(x + w / 2), int(y + h / 2)
                        crop = frame[y1:y2, x1:x2]
                        if crop.size > 0:
                            crop = cv2.resize(crop, tuple(self.config.get("cnn_input_size", [224, 224])))
                            crop = torch.from_numpy(crop).permute(2, 0, 1).float() / 255.0
                            crop = crop.unsqueeze(0).to(self.config.get("device", "cpu"))
                            with torch.no_grad():
                                output = self.cnn_model(crop)
                                state = torch.argmax(output, dim=1).item()
                                state_label = {0: "scoop", 1: "medium", 2: "neutral"}[state]

                    # Логирование в SQLite
                    c.execute(
                        "INSERT INTO events VALUES (?, ?, ?, ?, ?)",
                        (frame_file, state_label, conf, int(track_id), datetime.now().isoformat())
                    )
                    conn.commit()

                    # Подсчёт циклов
                    if state_label != "medium":
                        if self.last_state == "scoop" and state_label == "dump":
                            self.cycles += 1
                            output = self.cycles * self.bucket_volume
                            self.cycle_count.emit(self.cycles, output)
                            logging.info(f"Cycle detected: {self.cycles}, Output: {output:.2f} m³")
                        self.last_state = state_label

                # Сохранение аннотаций
                os.makedirs(f"{self.output_dir}/annotations", exist_ok=True)
                with open(f"{self.output_dir}/annotations/yolo.txt", "a", encoding='utf-8') as f:
                    for x, y, w, h, conf in frame_annotations:
                        f.write(f"{frame_file}: bucket ({x},{y},{w},{h}), conf: {conf:.2f}\n")

                self.progress.emit(int((i + 1) / total_frames * 100))

            # Сохранение отчёта
            self._save_report()
            self.status_signal.emit(f"Обработка завершена! Циклов: {self.cycles}, Выработка: {self.cycles * self.bucket_volume:.2f} м³")
            logging.info("Processing completed")
            conn.close()

        except Exception as e:
            self.status_signal.emit(f"Ошибка: {str(e)}")
            logging.error(f"Processing failed: {str(e)}")
        finally:
            self.finished.emit()

    def _save_report(self) -> None:
        """Сохраняет отчёт о циклах и выработке."""
        try:
            report_path = f"{self.output_dir}/report.txt"
            with open(report_path, "w", encoding='utf-8') as f:
                f.write(f"Циклов: {self.cycles}\n")
                f.write(f"Выработка: {self.cycles * self.bucket_volume:.2f} м³\n")
                f.write(f"Время обработки: {datetime.now().isoformat()}\n")
            logging.info(f"Report saved: {report_path}")
        except Exception as e:
            logging.error(f"Failed to save report: {str(e)}")