from typing import Optional
from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QLabel, QProgressBar, QComboBox
from PyQt5.QtCore import Qt
from core.interfaces.module import ModuleInterface
from core.config import Config
from operator_module.ui.result_viewer import ResultViewer
from operator_module.processor import OperatorProcessor
from core.models.yolo_model import YoloModel
from core.models.cnn import SimpleCNN
import torch
import logging
from datetime import datetime
import os

class OperatorModule(QMainWindow, ModuleInterface):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ZeroBlindSpot - Operator")
        self.setGeometry(100, 100, 800, 400)
        self.config = Config()
        self.yolo_model: Optional[YoloModel] = None
        self.cnn_model: Optional[SimpleCNN] = None
        self.device = self._get_device()
        self.excavators = {
            "Экскаватор A": 1.5,
            "Экскаватор B": 2.0,
            "Экскаватор C": 2.5
        }
        self._init_models()
        self._init_ui()

    def _init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)
        self.status_label = QLabel("Ожидание загрузки видео...")
        control_layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        control_layout.addWidget(self.progress_bar)

        self.excavator_combo = QComboBox()
        self.excavator_combo.addItems(self.excavators.keys())
        self.excavator_combo.currentTextChanged.connect(self._update_excavator)
        excavator = self.config.get("excavator", "Экскаватор A")
        self.excavator_combo.setCurrentText(excavator)
        control_layout.addWidget(self.excavator_combo)

        self.load_button = QPushButton("Загрузить видео")
        self.load_button.clicked.connect(self._load_video)
        control_layout.addWidget(self.load_button)

        main_layout.addWidget(control_widget, 1)

        self.result_viewer = ResultViewer()
        main_layout.addWidget(self.result_viewer, 3)

        try:
            with open("config/styles.qss", "r", encoding='utf-8') as f:
                self.setStyleSheet(f.read())
        except FileNotFoundError:
            logging.warning("styles.qss not found")

    def _init_models(self):
        try:
            self.yolo_model = YoloModel("models/model.pt")
            logging.info("YOLO loaded")
        except Exception as e:
            logging.error(f"YOLO load failed: {str(e)}")
            self.status_label.setText(f"Ошибка YOLO: {str(e)}")

        if self.config.get("use_cnn", False):
            try:
                self.cnn_model = SimpleCNN()
                self.cnn_model.load_state_dict(torch.load("models/bucket_cnn.pth", map_location=self.device, weights_only=True))
                self.cnn_model.to(self.device).eval()
                logging.info(f"CNN loaded on {self.device}")
            except Exception as e:
                logging.error(f"CNN load failed: {str(e)}")
                self.cnn_model = None

    def _get_device(self) -> torch.device:
        config_device = self.config.get("device", "auto")
        return torch.device("cuda" if config_device == "auto" and torch.cuda.is_available() else config_device)

    def _update_excavator(self, excavator: str):
        self.config.update("excavator", excavator)
        self.config.update("bucket_volume", self.excavators[excavator])
        logging.info(f"Excavator: {exavator}, Bucket volume: {self.excavators[excavator]} m³")

    def _load_video(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Выберите видео", "", "Video Files (*.mp4)")
        if file_name:
            self.status_label.setText(f"Загружено: {os.path.basename(file_name)}")
            self.progress_bar.setValue(0)
            self.result_viewer.reset()
            logging.info(f"Video: {file_name}")
            self._process_video(file_name)

    def _process_video(self, video_path: str):
        if not self.yolo_model:
            self.status_label.setText("Ошибка: YOLO не загружен")
            return
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        output_dir = f"data/{timestamp}"
        self.processor = OperatorProcessor(video_path, self.yolo_model, self.cnn_model, self.config, output_dir)
        self.processor.progress.connect(self.progress_bar.setValue)
        self.processor.status_signal.connect(self.status_label.setText)
        self.processor.cycle_count.connect(self.result_viewer.update_results)
        self.processor.finished.connect(self._on_finish)
        self.load_button.setEnabled(False)
        self.config.update("last_project", timestamp)
        self.processor.start()

    def _on_finish(self):
        self.load_button.setEnabled(True)

    def start(self):
        self.show()