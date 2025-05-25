import sys
import os
import cv2
import numpy as np
import torch
import yaml
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel, QComboBox, QProgressBar
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import ffmpeg
from ultralytics import YOLO
import logging

# Настройка логирования
logging.basicConfig(
    filename='logs/app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class VideoProcessor(QThread):
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, video_path, yolo_model, config):
        super().__init__()
        self.video_path = video_path
        self.yolo_model = yolo_model
        self.config = config

    def run(self):
        try:
            # Разбиение видео на кадры
            output_dir = "data/frames"
            os.makedirs(output_dir, exist_ok=True)
            frame_rate = self.config.get("frame_rate", 1)
            stream = ffmpeg.input(self.video_path)
            stream = ffmpeg.output(stream, f"{output_dir}/frame_%04d.jpg", r=frame_rate)
            ffmpeg.run(stream, overwrite_output=True)
            logging.info(f"Video split into frames in {output_dir}")

            frame_files = sorted(os.listdir(output_dir))
            if not frame_files:
                self.status.emit("Ошибка: кадры не созданы")
                logging.error("No frames created in data/frames")
                return

            # Обработка кадров
            total_frames = len(frame_files)
            for i, frame_file in enumerate(frame_files):
                frame_path = os.path.join(output_dir, frame_file)
                frame = cv2.imread(frame_path)
                if frame is None:
                    logging.warning(f"Failed to read frame: {frame_path}")
                    continue

                # YOLO: детекция ковша
                results = self.yolo_model(frame)
                boxes = results[0].boxes.xywh.cpu().numpy()
                confidences = results[0].boxes.conf.cpu().numpy()
                for box, conf in zip(boxes, confidences):
                    x, y, w, h = box
                    x1, y1 = int(x - w/2), int(y - h/2)
                    x2, y2 = int(x + w/2), int(y + h/2)
                    if x2 <= x1 or y2 <= y1:
                        logging.warning(f"Invalid box {box} in {frame_file}")
                        continue

                    # Сохранение аннотаций
                    os.makedirs("data/annotations", exist_ok=True)
                    with open("data/annotations/yolo.txt", "a") as f:
                        f.write(f"{frame_file}: bucket ({x},{y},{w},{h}), conf: {conf:.2f}\n")
                    logging.info(f"Processed frame {frame_file}: bucket, conf: {conf:.2f}")

                # Обновление прогресса
                self.progress.emit(int((i + 1) / total_frames * 100))

            self.status.emit("Обработка завершена!")
            logging.info("Video processing completed")
        except Exception as e:
            self.status.emit(f"Ошибка обработки: {str(e)}")
            logging.error(f"Video processing failed: {str(e)}")
        finally:
            self.finished.emit()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ZeroBlindSpot")
        self.setGeometry(100, 100, 800, 600)

        # Главный виджет и макет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Метка для статуса
        self.status_label = QLabel("Ожидание загрузки видео...")
        layout.addWidget(self.status_label)

        # Прогресс-бар
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        # Выбор устройства
        self.device_combo = QComboBox()
        self.device_combo.addItems(["auto", "cpu"])
        if torch.cuda.is_available():
            self.device_combo.addItem("cuda")
        self.device_combo.currentTextChanged.connect(self.update_device)
        layout.addWidget(self.device_combo)

        # Кнопка загрузки видео
        self.load_button = QPushButton("Загрузить видео")
        self.load_button.clicked.connect(self.load_video)
        layout.addWidget(self.load_button)

        # Загрузка конфигурации
        self.config = self.load_config()
        self.device = self.get_device()
        self.device_combo.setCurrentText(self.config.get("device", "auto"))

        # Инициализация моделей
        self.yolo_model = None
        self.cnn_model = None
        try:
            self.yolo_model = YOLO("models/model.pt")
            logging.info("YOLO model loaded successfully")
        except Exception as e:
            logging.error(f"Failed to load YOLO model: {str(e)}")
            self.status_label.setText(f"Ошибка загрузки YOLO: {str(e)}")

        # CNN временно отключена
        logging.info("CNN model disabled until architecture is provided")

        # Применение стилей
        try:
            with open("ui/styles.qss", "r") as f:
                self.setStyleSheet(f.read())
        except FileNotFoundError:
            logging.warning("styles.qss not found, using default styles")

    def load_config(self):
        config_path = "config/config.yaml"
        default_config = {
            "device": "auto",
            "frame_rate": 1,
            "cnn_input_size": [224, 224],
            "use_cnn": False
        }
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
                return config or default_config
        except FileNotFoundError:
            logging.warning("config.yaml not found, using default config")
            with open(config_path, "w") as f:
                yaml.safe_dump(default_config, f)
            return default_config

    def update_device(self, device):
        self.config["device"] = device
        with open("config/config.yaml", "w") as f:
            yaml.safe_dump(self.config, f)
        self.device = self.get_device()
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
            logging.info(f"Selected video: {file_name}")
            self.process_video(file_name)

    def process_video(self, video_path):
        if not self.yolo_model:
            self.status_label.setText("Ошибка: YOLO модель не загружена")
            logging.error("Cannot process video: YOLO model not loaded")
            return

        self.processor = VideoProcessor(video_path, self.yolo_model, self.config)
        self.processor.progress.connect(self.progress_bar.setValue)
        self.processor.status.connect(self.status_label.setText)
        self.processor.finished.connect(self.on_processing_finished)
        self.processor.start()

    def on_processing_finished(self):
        self.load_button.setEnabled(True)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())