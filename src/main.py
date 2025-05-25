import sys
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import yaml
import shutil
from datetime import datetime
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QLabel, QComboBox, QProgressBar, QScrollArea
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QRect
from PyQt5.QtGui import QImage, QPixmap, QPainter
import ffmpeg
from ultralytics import YOLO
import logging
from logging.handlers import TimedRotatingFileHandler
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QRect, QPoint
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog, QScrollArea, QComboBox, QProgressBar



# Настройка логирования
def setup_logging(project_name="default"):
    try:
        log_dir = f"logs/{project_name}"
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        log_file = f"{log_dir}/app_{timestamp}.log"
        handler = TimedRotatingFileHandler(log_file, when="midnight", backupCount=30, encoding='utf-8')
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
        logging.info("Logging initialized successfully")
    except Exception as e:
        print(f"Failed to setup logging: {str(e)}")

# CNN модель
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, 3)  # scoop=0, medium=1, neutral=2

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 64 * 28 * 28)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class VideoProcessor(QThread):
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished = pyqtSignal()
    frame_processed = pyqtSignal(str, list)  # frame_file, [(x, y, w, h, conf)]
    no_bucket_frame = pyqtSignal(str)  # frame without bucket
    low_conf_frame = pyqtSignal(str, list)  # frame with low confidence

    def __init__(self, video_path, yolo_model, cnn_model, config, output_dir):
        super().__init__()
        self.video_path = video_path
        self.yolo_model = yolo_model
        self.cnn_model = cnn_model
        self.config = config
        self.output_dir = output_dir

    def run(self):
        try:
            # Разбиение видео на кадры
            self.status.emit("Разбиение видео на кадры...")
            frame_dir = f"{self.output_dir}/frames"
            no_bucket_dir = f"{self.output_dir}/no_bucket"
            os.makedirs(frame_dir, exist_ok=True)
            os.makedirs(no_bucket_dir, exist_ok=True)
            frame_rate = self.config.get("frame_rate", 1)
            stream = ffmpeg.input(self.video_path)
            stream = ffmpeg.output(stream, f"{frame_dir}/frame_%04d.jpg", r=frame_rate)
            ffmpeg.run(stream, overwrite_output=True)
            logging.info(f"Video split into frames in {frame_dir}")

            frame_files = sorted(os.listdir(frame_dir))
            if not frame_files:
                self.status.emit("Ошибка: кадры не созданы")
                logging.error(f"No frames created in {frame_dir}")
                return

            # Обработка кадров
            self.status.emit("Обработка кадров...")
            total_frames = len(frame_files)
            for i, frame_file in enumerate(frame_files):
                frame_path = os.path.join(frame_dir, frame_file)
                frame = cv2.imread(frame_path)
                if frame is None:
                    logging.warning(f"Failed to read frame: {frame_path}")
                    continue

                # YOLO: детекция ковша
                results = self.yolo_model(frame)
                boxes = results[0].boxes.xywh.cpu().numpy()
                confidences = results[0].boxes.conf.cpu().numpy()
                frame_annotations = []
                low_conf = False
                for box, conf in zip(boxes, confidences):
                    if conf < self.config.get("yolo_conf_threshold", 0.5):
                        continue
                    x, y, w, h = box
                    x1, y1 = int(x - w/2), int(y - h/2)
                    x2, y2 = int(x + w/2), int(y + h/2)
                    if x2 <= x1 or y2 <= y1:
                        logging.warning(f"Invalid box {box} in {frame_file}")
                        continue

                    frame_annotations.append((x, y, w, h, conf))
                    if conf < 0.6:
                        low_conf = True
                    # Сохранение аннотаций YOLO
                    os.makedirs(f"{self.output_dir}/annotations", exist_ok=True)
                    with open(f"{self.output_dir}/annotations/yolo.txt", "a", encoding='utf-8') as f:
                        f.write(f"{frame_file}: bucket ({x},{y},{w},{h}), conf: {conf:.2f}\n")
                    logging.info(f"Processed frame {frame_file}: bucket, conf: {conf:.2f}")

                    # CNN: классификация ковша
                    # Внутри цикла обработки кадров, после YOLO:
                    # Внутри цикла обработки кадров:
                    logging.info(f"Processing frame: {frame_file}")
                    state = 2
                    state_label = "neutral"
                    if self.config.get("use_cnn", False) and self.cnn_model and frame_annotations:
                        for box, conf in zip(boxes, confidences):
                            x, y, w, h = box
                            x1, y1 = int(x - w/2), int(y - h/2)
                            x2, y2 = int(x + w/2), int(y + h/2)
                            try:
                                crop = frame[y1:y2, x1:x2]
                                if crop.size == 0:
                                    logging.warning(f"Empty crop for box {box} in {frame_file}")
                                    continue
                                crop = cv2.resize(crop, tuple(self.config.get("cnn_input_size", [224, 224])))
                                crop = torch.from_numpy(crop).permute(2, 0, 1).float() / 255.0
                                crop = crop.unsqueeze(0).to(self.config["device"])
                                with torch.no_grad():
                                    output = self.cnn_model(crop)
                                    state = torch.argmax(output, dim=1).item()
                                    state_label = {0: "scoop", 1: "medium", 2: "neutral"}[state]
                                logging.info(f"CNN classified {frame_file}: {state_label}")
                            except Exception as e:
                                logging.error(f"CNN failed for {frame_file}: {str(e)}")
                                continue
                    os.makedirs(f"{self.output_dir}/annotations", exist_ok=True)
                    cnn_file = f"{self.output_dir}/annotations/cnn.csv"
                    mode = "a" if os.path.exists(cnn_file) else "w"
                    with open(cnn_file, mode, encoding='utf-8') as f:
                        if mode == "w":
                            f.write("frame,state,state_label\n")
                        f.write(f"{frame_file},{state},{state_label}\n")
                    logging.info(f"Wrote to cnn.csv: {frame_file},{state},{state_label}")
                    
                # Кадры без детекции
                if not frame_annotations:
                    no_bucket_path = os.path.join(no_bucket_dir, frame_file)
                    shutil.move(frame_path, no_bucket_path)
                    self.no_bucket_frame.emit(no_bucket_path)
                    logging.info(f"Moved frame {frame_file} to no_bucket")
                    continue
                # Кадры с низкой уверенностью
                if low_conf:
                    self.low_conf_frame.emit(frame_path, frame_annotations)

                self.frame_processed.emit(frame_path, frame_annotations)
                self.progress.emit(int((i + 1) / total_frames * 100))

            self.status.emit("Обработка завершена!")
            logging.info("Video processing completed")
        except Exception as e:
            self.status.emit(f"Ошибка обработки: {str(e)}")
            logging.error(f"Video processing failed: {str(e)}")
        finally:
            self.finished.emit()

class FrameViewer(QWidget):
    update_status = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.image_label = QLabel("Нет кадра")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.image_label)
        self.annotation_label = QLabel("Аннотации: отсутствуют")
        self.layout.addWidget(self.annotation_label)
        self.current_frame_path = None
        self.current_annotations = []
        self.all_annotations = {}  # {frame_path: [(x, y, w, h, conf), ...]}
        self.annotation_mode = False
        self.review_mode = False
        self.manual_review_mode = False
        self.no_bucket_frames = []
        self.low_conf_frames = []
        self.current_frame_index = -1

        # Панель управления
        self.control_layout = QHBoxLayout()
        self.prev_button = QPushButton("Предыдущий кадр")
        self.prev_button.clicked.connect(self.prev_frame)
        self.prev_button.setEnabled(False)
        self.control_layout.addWidget(self.prev_button)
        self.next_button = QPushButton("Следующий кадр")
        self.next_button.clicked.connect(self.next_frame)
        self.next_button.setEnabled(False)
        self.control_layout.addWidget(self.next_button)
        self.delete_button = QPushButton("Удалить кадр")
        self.delete_button.clicked.connect(self.delete_frame)
        self.delete_button.setEnabled(False)
        self.control_layout.addWidget(self.delete_button)
        self.correct_button = QPushButton("Верно")
        self.correct_button.clicked.connect(self.mark_correct)
        self.correct_button.setEnabled(False)
        self.control_layout.addWidget(self.correct_button)
        self.error_button = QPushButton("Ошибка")
        self.error_button.clicked.connect(self.mark_error)
        self.error_button.setEnabled(False)
        self.control_layout.addWidget(self.error_button)
        self.undo_button = QPushButton("Отмена")
        self.undo_button.clicked.connect(self.undo_annotation)
        self.undo_button.setEnabled(False)
        self.control_layout.addWidget(self.undo_button)
        self.save_button = QPushButton("Сохранить аннотации")
        self.save_button.clicked.connect(self.save_annotations)
        self.save_button.setEnabled(False)
        self.control_layout.addWidget(self.save_button)
        self.layout.addLayout(self.control_layout)

    def annotate_frame(self):
        if not self.current_frame_path or not self.annotation_mode:
            return
        frame = cv2.imread(self.current_frame_path)
        if frame is None:
            logging.error(f"Failed to read frame: {self.current_frame_path}")
            return
        temp_frame = frame.copy()
        start_point = None
        drawing = False
        annotations = self.all_annotations.get(self.current_frame_path, []).copy()

        def mouse_callback(event, x, y, flags, param):
            nonlocal start_point, drawing, temp_frame, annotations
            if event == cv2.EVENT_LBUTTONDOWN:
                start_point = (x, y)
                drawing = True
                logging.info(f"Annotation started at ({x}, {y})")
            elif event == cv2.EVENT_MOUSEMOVE and drawing:
                temp_frame = frame.copy()
                for ann in annotations:
                    ax, ay, aw, ah, _ = ann
                    x1, y1 = int(ax - aw/2), int(ay - ah/2)
                    x2, y2 = int(ax + aw/2), int(ay + ah/2)
                    cv2.rectangle(temp_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(temp_frame, start_point, (x, y), (0, 255, 0), 2)
                cv2.imshow("Annotate", temp_frame)
            elif event == cv2.EVENT_LBUTTONUP and drawing:
                drawing = False
                x1, y1 = start_point
                x2, y2 = x, y
                x = (x1 + x2) / 2
                y = (y1 + y2) / 2
                w = abs(x2 - x1)
                h = abs(y2 - y1)
                if w > 5 and h > 5:
                    annotations.append((x, y, w, h, 1.0))
                    self.all_annotations[self.current_frame_path] = annotations
                    self.current_annotations = annotations
                    self.save_annotations()
                    logging.info(f"Added annotation: ({x}, {y}, {w}, {h})")
                temp_frame = frame.copy()
                for ann in annotations:
                    ax, ay, aw, ah, _ = ann
                    x1, y1 = int(ax - aw/2), int(ay - ah/2)
                    x2, y2 = int(ax + aw/2), int(ay + ah/2)
                    cv2.rectangle(temp_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.imshow("Annotate", temp_frame)

        cv2.namedWindow("Annotate")
        cv2.setMouseCallback("Annotate", mouse_callback)
        cv2.imshow("Annotate", temp_frame)
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                cv2.destroyAllWindows()
                break
            if key == ord("s"):
                self.all_annotations[self.current_frame_path] = annotations
                self.current_annotations = annotations
                self.save_annotations()
                cv2.destroyAllWindows()
                break
            if key == ord("a"):
                self.all_annotations[self.current_frame_path] = annotations
                self.current_annotations = annotations
                self.save_annotations()
                cv2.destroyAllWindows()
                self.prev_frame()
                break
            if key == ord("d"):
                self.all_annotations[self.current_frame_path] = annotations
                self.current_annotations = annotations
                self.save_annotations()
                cv2.destroyAllWindows()
                self.next_frame()
                break
        self.update_frame(self.current_frame_path, self.current_annotations)

    def update_frame(self, frame_path, annotations):
        if not frame_path:
            self.image_label.setPixmap(QPixmap())
            self.annotation_label.setText("Аннотации: отсутствуют")
            self.current_frame_path = None
            self.current_annotations = []
            self.delete_button.setEnabled(False)
            self.correct_button.setEnabled(False)
            self.error_button.setEnabled(False)
            self.undo_button.setEnabled(False)
            self.save_button.setEnabled(False)
            return

        self.current_frame_path = frame_path
        self.current_annotations = annotations
        self.all_annotations[frame_path] = annotations  # Сохранить
        frame = cv2.imread(frame_path)
        if frame is None:
            logging.error(f"Failed to read frame: {frame_path}")
            return
        for x, y, w, h, conf in annotations:
            x1, y1 = int(x - w/2), int(y - h/2)
            x2, y2 = int(x + w/2), int(y + h/2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"bucket: {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        q_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(q_image).scaled(640, 480, Qt.KeepAspectRatio))
        self.annotation_label.setText(f"Аннотации: {len(annotations)} box(es)")

        frames = self.no_bucket_frames if self.annotation_mode else [f[0] for f in self.low_conf_frames]
        self.prev_button.setEnabled(self.current_frame_index > 0)
        self.next_button.setEnabled(self.current_frame_index < len(frames) - 1)
        self.delete_button.setEnabled(bool(self.current_frame_path) and self.annotation_mode)
        self.correct_button.setEnabled(bool(self.current_frame_path) and self.review_mode)
        self.error_button.setEnabled(bool(self.current_frame_path) and self.review_mode)
        self.undo_button.setEnabled(bool(self.current_frame_path) and self.annotation_mode and len(self.current_annotations) > 0)
        self.save_button.setEnabled(bool(self.current_frame_path) and self.annotation_mode and len(self.current_annotations) > 0)
        if self.annotation_mode:
            self.annotate_frame()

    def add_no_bucket_frame(self, frame_path):
        self.no_bucket_frames.append(frame_path)
        logging.info(f"Added no-bucket frame: {frame_path}")

    def add_low_conf_frame(self, frame_path, annotations):
        self.low_conf_frames.append((frame_path, annotations))
        logging.info(f"Added low-conf frame: {frame_path}")

    def load_manual_frames(self):
        output_dir = os.path.dirname(os.path.dirname(self.no_bucket_frames[0])) if self.no_bucket_frames else ""
        if not output_dir:
            return
        yolo_file = os.path.join(output_dir, "annotations", "yolo.txt")
        frame_dir = os.path.join(output_dir, "frames")
        self.low_conf_frames = []
        manual_frames = {}
        if os.path.exists(yolo_file):
            with open(yolo_file, "r", encoding='utf-8') as f:
                for line in f:
                    if "(manual)" in line:
                        frame_file = line.split(":")[0].strip()
                        parts = line.split(": bucket (")[1].split(", conf: ")
                        coords = parts[0].strip("()").split(",")
                        x, y, w, h = map(float, coords)
                        conf = float(parts[1].split()[0])
                        frame_path = os.path.join(frame_dir, frame_file)
                        if frame_file not in manual_frames:
                            manual_frames[frame_file] = []
                        manual_frames[frame_file].append((x, y, w, h, conf))
        for frame_file, annotations in manual_frames.items():
            frame_path = os.path.join(frame_dir, frame_file)
            if os.path.exists(frame_path):
                self.low_conf_frames.append((frame_path, annotations))
        logging.info(f"Loaded {len(self.low_conf_frames)} manual annotation frames")

    def show_frame(self, index, is_annotation_mode):
        frames = self.no_bucket_frames if is_annotation_mode else [f[0] for f in self.low_conf_frames]
        annotations = [] if is_annotation_mode else self.low_conf_frames[index][1] if index < len(self.low_conf_frames) else []
        if 0 <= index < len(frames):
            self.current_frame_index = index
            frame_path = frames[index]
            output_dir = os.path.dirname(os.path.dirname(frame_path))
            yolo_file = os.path.join(output_dir, "annotations", "yolo.txt")
            annotations = self.all_annotations.get(frame_path, [])
            if os.path.exists(yolo_file):
                frame_file = os.path.basename(frame_path)
                with open(yolo_file, "r", encoding='utf-8') as f:
                    for line in f:
                        if frame_file in line and ": bucket" in line and "(manual)" in line:
                            parts = line.split(": bucket (")[1].split(", conf: ")
                            coords = parts[0].strip("()").split(",")
                            x, y, w, h = map(float, coords)
                            conf = float(parts[1].split()[0])
                            if not any(a[:4] == (x, y, w, h) for a in annotations):
                                annotations.append((x, y, w, h, conf))
            self.all_annotations[frame_path] = annotations
            self.update_frame(frame_path, annotations)
            status = f"Кадр {index + 1}/{len(frames)} {'без ковша' if is_annotation_mode else 'с низкой уверенностью'}. Фреймов без детекции: {len(self.no_bucket_frames)}"
            if self.manual_review_mode:
                status = f"Кадр {index + 1}/{len(frames)} с ручной аннотацией. Фреймов без детекции: {len(self.no_bucket_frames)}"
            self.update_status.emit(status)
        else:
            self.update_frame("", [])
            self.update_status.emit(f"Нет кадров {'без ковша' if is_annotation_mode else 'с низкой уверенностью'}. Фреймов без детекции: {len(self.no_bucket_frames)}")

    def prev_frame(self):
        if self.current_frame_path and self.annotation_mode and self.current_annotations:
            self.save_annotations()
        new_index = self.current_frame_index - 1
        if new_index >= 0:
            self.show_frame(new_index, self.annotation_mode)
            logging.info(f"Previous frame, index: {new_index}")
        else:
            self.update_status.emit("Нет предыдущих кадров")

    def next_frame(self):
        if self.current_frame_path and self.annotation_mode and self.current_annotations:
            self.save_annotations()
        frames = self.no_bucket_frames if self.annotation_mode else [f[0] for f in self.low_conf_frames]
        if self.current_frame_index + 1 < len(frames):
            self.current_frame_index += 1
            self.show_frame(self.current_frame_index, self.annotation_mode)
            logging.info(f"Next frame, index: {self.current_frame_index}, total: {len(frames)}")
        else:
            self.current_frame_index = -1
            self.update_frame("", [])
            self.update_status.emit(f"Проверка завершена, кадров: {len(frames)}")
            logging.info(f"Review completed, total frames: {len(frames)}")

    def delete_frame(self):
        if self.current_frame_path and self.annotation_mode and self.current_frame_index >= 0:
            frame_file = os.path.basename(self.current_frame_path)
            os.remove(self.current_frame_path)
            logging.info(f"Manually deleted frame: {frame_file}")
            del self.no_bucket_frames[self.current_frame_index]
            if self.no_bucket_frames:
                self.show_frame(min(self.current_frame_index, len(self.no_bucket_frames) - 1), True)
            else:
                self.show_frame(-1, True)

    def undo_annotation(self):
        if self.current_annotations:
            self.current_annotations.pop()
            logging.info("Removed last annotation")
            self.update_frame(self.current_frame_path, self.current_annotations)
            frame_file = os.path.basename(self.current_frame_path)
            output_dir = os.path.dirname(os.path.dirname(self.current_frame_path))
            with open(f"{output_dir}/annotations/yolo.txt", "a", encoding='utf-8') as f:
                f.write(f"{frame_file}: removed last annotation\n")
        self.undo_button.setEnabled(len(self.current_annotations) > 0)

    def save_annotations(self):
        if self.current_frame_path and self.current_annotations:
            frame_file = os.path.basename(self.current_frame_path)
            output_dir = os.path.dirname(os.path.dirname(self.current_frame_path))
            frame_dir = os.path.join(output_dir, "frames")
            no_bucket_dir = os.path.join(output_dir, "no_bucket")
            os.makedirs(f"{output_dir}/annotations", exist_ok=True)
            os.makedirs(frame_dir, exist_ok=True)
            # Сохранить аннотации
            with open(f"{output_dir}/annotations/yolo.txt", "a", encoding='utf-8') as f:
                for x, y, w, h, conf in self.current_annotations:
                    f.write(f"{frame_file}: bucket ({x},{y},{w},{h}), conf: {conf:.2f} (manual)\n")
            # Переместить фрейм
            if self.current_frame_path.startswith(no_bucket_dir):
                new_path = os.path.join(frame_dir, frame_file)
                shutil.move(self.current_frame_path, new_path)
                self.current_frame_path = new_path
                if self.current_frame_index < len(self.no_bucket_frames):
                    self.no_bucket_frames[self.current_frame_index] = new_path
                else:
                    self.no_bucket_frames.append(new_path)
                logging.info(f"Moved {frame_file} to frames for training")
            self.all_annotations[self.current_frame_path] = self.current_annotations
            logging.info(f"Saved annotations for {frame_file}: {len(self.current_annotations)} boxes")
            self.update_status.emit(f"Аннотации сохранены для {frame_file}. Фреймов без детекции: {len(self.no_bucket_frames)}")

    def mark_correct(self):
        if self.current_frame_path and self.review_mode:
            frame_file = os.path.basename(self.current_frame_path)
            output_dir = os.path.dirname(os.path.dirname(self.current_frame_path))
            frame_dir = os.path.join(output_dir, "frames")
            no_bucket_dir = os.path.join(output_dir, "no_bucket")
            os.makedirs(f"{output_dir}/annotations", exist_ok=True)
            os.makedirs(frame_dir, exist_ok=True)
            # Переместить из no_bucket
            if self.current_frame_path.startswith(no_bucket_dir):
                new_path = os.path.join(frame_dir, frame_file)
                shutil.move(self.current_frame_path, new_path)
                self.current_frame_path = new_path
                logging.info(f"Moved {frame_file} from no_bucket to frames")
            # Обновить yolo.txt
            with open(f"{output_dir}/annotations/yolo.txt", "a", encoding='utf-8') as f:
                for x, y, w, h, conf in self.current_annotations:
                    f.write(f"{frame_file}: bucket ({x},{y},{w},{h}), conf: {conf:.2f} (confirmed)\n")
            # Сохранить в review.txt
            with open(f"{output_dir}/annotations/review.txt", "a", encoding='utf-8') as f:
                f.write(f"{frame_file}: correct, boxes: {self.current_annotations}\n")
            logging.info(f"Marked {frame_file} as correct with {len(self.current_annotations)} boxes, index: {self.current_frame_index}")
            # Удалить из low_conf_frames
            if self.low_conf_frames and self.current_frame_index < len(self.low_conf_frames):
                del self.low_conf_frames[self.current_frame_index]
                self.current_frame_index -= 1  # Компенсировать удаление
            self.next_frame()

    def mark_error(self):
        if self.current_frame_path and self.review_mode:
            frame_file = os.path.basename(self.current_frame_path)
            output_dir = os.path.dirname(os.path.dirname(self.current_frame_path))
            deleted_dir = os.path.join(output_dir, "deleted_frames")
            os.makedirs(f"{output_dir}/annotations", exist_ok=True)
            os.makedirs(deleted_dir, exist_ok=True)
            # Переместить в deleted_frames
            new_path = os.path.join(deleted_dir, frame_file)
            shutil.move(self.current_frame_path, new_path)
            with open(f"{output_dir}/annotations/review.txt", "a", encoding='utf-8') as f:
                f.write(f"{frame_file}: error, boxes: {self.current_annotations}\n")
            logging.info(f"Marked {frame_file} as error, moved to deleted_frames, index: {self.current_frame_index}")
            # Удалить из low_conf_frames
            if self.low_conf_frames and self.current_frame_index < len(self.low_conf_frames):
                del self.low_conf_frames[self.current_frame_index]
                self.current_frame_index -= 1  # Компенсировать удаление
            self.next_frame()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ZeroBlindSpot")
        self.setGeometry(100, 100, 1000, 600)

        # Инициализация моделей
        self.yolo_model = None
        self.cnn_model = None

        # Главный виджет и макет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Левая панель: управление
        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)
        self.status_label = QLabel("Ожидание загрузки видео...")
        control_layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        control_layout.addWidget(self.progress_bar)

        self.device_combo = QComboBox()
        self.device_combo.addItems(["auto", "cpu"])
        if torch.cuda.is_available():
            self.device_combo.addItem("cuda")
        self.device_combo.currentTextChanged.connect(self.update_device)
        control_layout.addWidget(self.device_combo)

        self.load_button = QPushButton("Загрузить видео")
        self.load_button.clicked.connect(self.load_video)
        control_layout.addWidget(self.load_button)

        self.project_button = QPushButton("Выбрать проект")
        self.project_button.clicked.connect(self.load_project)
        control_layout.addWidget(self.project_button)

        self.annotate_button = QPushButton("Режим аннотации")
        self.annotate_button.setEnabled(False)
        self.annotate_button.clicked.connect(self.toggle_annotation)
        control_layout.addWidget(self.annotate_button)

        self.review_button = QPushButton("Проверка детекций")
        self.review_button.setEnabled(False)
        self.review_button.clicked.connect(self.toggle_review)
        control_layout.addWidget(self.review_button)

        self.manual_review_button = QPushButton("Проверить ручные аннотации")
        self.manual_review_button.clicked.connect(self.toggle_manual_review)
        self.manual_review_button.setEnabled(False)
        control_layout.addWidget(self.manual_review_button)  # Добавлено в control_layout

        main_layout.addWidget(control_widget, 1)

        # Правая панель: просмотр кадров
        self.frame_viewer = FrameViewer()
        self.frame_viewer.update_status.connect(self.status_label.setText)
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.frame_viewer)
        scroll_area.setWidgetResizable(True)
        main_layout.addWidget(scroll_area, 3)

        # Загрузка конфигурации
        self.config = self.load_config()
        self.device = self.get_device()
        self.device_combo.setCurrentText(self.config.get("device", "auto"))

        # Инициализация YOLO
        try:
            self.yolo_model = YOLO("models/model.pt")
            logging.info("YOLO model loaded successfully")
        except Exception as e:
            logging.error(f"Failed to load YOLO model: {str(e)}")
            self.status_label.setText(f"Ошибка загрузки YOLO: {str(e)}")

        # Инициализация CNN
        if self.config.get("use_cnn", False):
            try:
                self.cnn_model = SimpleCNN()
                state_dict = torch.load(
                    "models/bucket_cnn.pth",
                    map_location=self.device,
                    weights_only=True
                )
                model_keys = set(self.cnn_model.state_dict().keys())
                loaded_keys = set(state_dict.keys())
                missing = model_keys - loaded_keys
                extra = loaded_keys - model_keys
                logging.info(f"CNN model keys: {model_keys}")
                logging.info(f"Loaded keys: {loaded_keys}")
                if missing or extra:
                    logging.warning(f"CNN keys mismatch: missing={missing}, extra={extra}")
                self.cnn_model.load_state_dict(state_dict, strict=False)
                self.cnn_model.to(self.device).eval()
                logging.info(f"CNN model loaded on {self.device}")
            except Exception as e:
                logging.error(f"Failed to load CNN model: {str(e)}")
                self.status_label.setText(f"Ошибка загрузки CNN: {str(e)}")
                self.cnn_model = None
                logging.warning("CNN disabled due to load failure")
        else:
            logging.info("CNN model disabled")

        # Применение стилей
        try:
            with open("ui/styles.qss", "r", encoding='utf-8') as f:
                self.setStyleSheet(f.read())
        except FileNotFoundError:
            logging.warning("styles.qss not found, using default styles")

    def load_config(self):
        config_path = "config/config.yaml"
        default_config = {
            "device": "auto",
            "frame_rate": 1,
            "cnn_input_size": [224, 224],
            "use_cnn": True,
            "yolo_conf_threshold": 0.5,
            "project_name": "default",
            "last_project": ""
        }
        try:
            with open(config_path, "r", encoding='utf-8') as f:
                config = yaml.safe_load(f)
                return config or default_config
        except FileNotFoundError:
            logging.warning("config.yaml not found, using default config")
            with open(config_path, "w", encoding='utf-8') as f:
                yaml.safe_dump(default_config, f, allow_unicode=True)
            return default_config

    def update_device(self, device):
        self.config["device"] = device
        with open("config/config.yaml", "w", encoding='utf-8') as f:
            yaml.safe_dump(self.config, f, allow_unicode=True)
        self.device = self.get_device()
        if hasattr(self, 'cnn_model') and self.cnn_model is not None:
            self.cnn_model.to(self.device)
        logging.info(f"Device updated to {device}")

    def get_device(self):
        config_device = self.config.get("device", "auto")
        if config_device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(config_device)

    def load_video(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Выберите видео", "", "Video Files (*.mp4 *.avi)")
        if file_name:
            self.status_label.setText(f"Загружено: {file_name}")
            self.progress_bar.setValue(0)
            self.frame_viewer.update_frame("", [])
            logging.info(f"Selected video: {file_name}")
            self.process_video(file_name)

    def load_project(self):
        cv2.destroyAllWindows()  # Закрыть OpenCV
        project_dir = QFileDialog.getExistingDirectory(self, "Выберите папку проекта", "data")
        if project_dir:
            self.status_label.setText(f"Загружен проект: {project_dir}")
            logging.info(f"Loaded project: {project_dir}")
            self.frame_viewer.no_bucket_frames = []
            self.frame_viewer.low_conf_frames = []
            self.frame_viewer.current_frame_index = -1
            self.frame_viewer.all_annotations = {}  # Сброс аннотаций
            frame_dir = os.path.join(project_dir, "frames")
            no_bucket_dir = os.path.join(project_dir, "no_bucket")
            deleted_dir = os.path.join(project_dir, "deleted_frames")
            review_file = os.path.join(project_dir, "annotations", "review.txt")
            reviewed_frames = set()
            if os.path.exists(review_file):
                with open(review_file, "r", encoding='utf-8') as f:
                    for line in f:
                        frame_file = line.split(":")[0].strip()
                        reviewed_frames.add(frame_file)
            # Загрузить no_bucket
            if os.path.exists(no_bucket_dir):
                for frame_file in sorted(os.listdir(no_bucket_dir)):
                    if frame_file not in reviewed_frames:
                        self.frame_viewer.add_no_bucket_frame(os.path.join(no_bucket_dir, frame_file))
            # Загрузить frames
            if os.path.exists(frame_dir):
                for frame_file in sorted(os.listdir(frame_dir)):
                    if frame_file not in reviewed_frames:
                        frame_path = os.path.join(frame_dir, frame_file)
                        annotations = []
                        yolo_file = os.path.join(project_dir, "annotations", "yolo.txt")
                        if os.path.exists(yolo_file):
                            with open(yolo_file, "r", encoding='utf-8') as f:
                                for line in f:
                                    if frame_file in line and ": bucket" in line and "(confirmed)" not in line:
                                        parts = line.split(": bucket (")[1].split(", conf: ")
                                        coords = parts[0].strip("()").split(",")
                                        x, y, w, h = map(float, coords)
                                        conf = float(parts[1].split()[0])
                                        annotations.append((x, y, w, h, conf))
                        if annotations and any(conf < 0.6 for _, _, _, _, conf in annotations):
                            self.frame_viewer.add_low_conf_frame(frame_path, annotations)
                        self.frame_viewer.update_frame(frame_path, annotations)
            self.annotate_button.setEnabled(True)
            self.review_button.setEnabled(True)
            self.config["last_project"] = project_dir
            with open("config/config.yaml", "w", encoding='utf-8') as f:
                yaml.safe_dump(self.config, f, allow_unicode=True)
            logging.info(f"Loaded {len(self.frame_viewer.low_conf_frames)} low-conf frames, {len(self.frame_viewer.no_bucket_frames)} no-bucket frames")
            self.manual_review_button.setEnabled(True)

    def process_video(self, video_path):
        if not self.yolo_model:
            self.status_label.setText("Ошибка: YOLO модель не загружена")
            logging.error("Cannot process video: YOLO model not loaded")
            return

        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        output_dir = f"data/{timestamp}"
        self.frame_viewer.no_bucket_frames = []
        self.frame_viewer.low_conf_frames = []
        self.frame_viewer.current_frame_index = -1
        self.processor = VideoProcessor(video_path, self.yolo_model, self.cnn_model, self.config, output_dir)
        self.processor.progress.connect(self.progress_bar.setValue)
        self.processor.status.connect(self.status_label.setText)
        self.processor.frame_processed.connect(self.frame_viewer.update_frame)
        self.processor.no_bucket_frame.connect(self.frame_viewer.add_no_bucket_frame)
        self.processor.low_conf_frame.connect(self.frame_viewer.add_low_conf_frame)
        self.processor.finished.connect(self.on_processing_finished)
        self.load_button.setEnabled(False)
        self.annotate_button.setEnabled(True)
        self.review_button.setEnabled(True)
        self.config["last_project"] = output_dir
        with open("config/config.yaml", "w", encoding='utf-8') as f:
            yaml.safe_dump(self.config, f, allow_unicode=True)
        self.processor.start()

    def toggle_annotation(self):
        self.frame_viewer.review_mode = False
        self.frame_viewer.annotation_mode = not self.frame_viewer.annotation_mode
        status = "включён" if self.frame_viewer.annotation_mode else "выключён"
        self.status_label.setText(f"Режим аннотации: {status}")
        logging.info(f"Annotation mode {status}")
        if self.frame_viewer.annotation_mode and self.frame_viewer.no_bucket_frames:
            self.frame_viewer.show_frame(0, True)
            self.frame_viewer.annotate_frame()
        elif self.frame_viewer.annotation_mode:
            self.status_label.setText("Режим аннотации: нет кадров без ковша")
        else:
            cv2.destroyAllWindows()  # Закрыть OpenCV

    def toggle_review(self):
        self.frame_viewer.annotation_mode = False
        self.frame_viewer.review_mode = not self.frame_viewer.review_mode
        status = "включён" if self.frame_viewer.review_mode else "выключён"
        self.status_label.setText(f"Режим проверки детекций: {status}")
        logging.info(f"Review mode {status}")
        if self.frame_viewer.review_mode and self.frame_viewer.low_conf_frames:
            self.frame_viewer.show_frame(0, False)
        elif self.frame_viewer.review_mode:
            self.status_label.setText("Режим проверки: нет кадров с низкой уверенностью")

    def toggle_manual_review(self):
        self.frame_viewer.annotation_mode = False
        self.frame_viewer.review_mode = not self.frame_viewer.review_mode
        self.frame_viewer.manual_review_mode = self.frame_viewer.review_mode
        status = "включён" if self.frame_viewer.review_mode else "выключён"
        self.status_label.setText(f"Режим проверки ручных аннотаций: {status}")
        logging.info(f"Manual review mode {status}")
        if self.frame_viewer.review_mode:
            self.frame_viewer.load_manual_frames()
            if self.frame_viewer.low_conf_frames:
                self.frame_viewer.show_frame(0, False)
            else:
                self.status_label.setText("Режим проверки ручных аннотаций: нет кадров")
        else:
            cv2.destroyAllWindows()

    def on_processing_finished(self):
        self.load_button.setEnabled(True)

if __name__ == "__main__":
    setup_logging()
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())