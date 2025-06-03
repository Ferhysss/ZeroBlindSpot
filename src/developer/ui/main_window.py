from typing import Optional
from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QLabel, QComboBox, QProgressBar
from PyQt5.QtCore import Qt
from core.interfaces.module import ModuleInterface
from core.config import Config
from developer.ui.frame_viewer import FrameViewer
from developer.processor import DeveloperProcessor
from core.models.yolo import YoloModel
from core.models.cnn import SimpleCNN
import torch
import logging

class DeveloperModule(QMainWindow, ModuleInterface):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ZeroBlindSpot - Developer")
        self.setGeometry(100, 100, 1000, 600)
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

        # Левая панель
        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)
        self.status_label = QLabel("Ожидание загрузки видео...")
        control_layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        control_layout.addWidget(self.progress_bar)

        # Выбор устройства
        self.device_combo = QComboBox()
        self.device_combo.addItems(["auto", "cpu"])
        if torch.cuda.is_available():
            self.device_combo.addItem("cuda")
        self.device_combo.currentTextChanged.connect(self._update_device)
        control_layout.addWidget(self.device_combo)

        # Выбор экскаватора
        self.excavator_combo = QComboBox()
        self.excavator_combo.addItems(self.excavators.keys())
        self.excavator_combo.currentTextChanged.connect(self._update_excavator)
        excavator = self.config.get("excavator", "Экскаватор A")
        self.excavator_combo.setCurrentText(excavator)
        control_layout.addWidget(self.excavator_combo)

        self.load_button = QPushButton("Загрузить видео")
        self.load_button.clicked.connect(self._load_video)
        control_layout.addWidget(self.load_button)

        self.extract_button = QPushButton("Извлечь кадры")
        self.extract_button.clicked.connect(self._extract_frames)
        control_layout.addWidget(self.extract_button)

        self.class_combo = QComboBox()
        self.class_combo.addItems(["bucket"])  # Можно добавить другие классы позже
        self.class_combo.currentTextChanged.connect(self._update_class)
        control_layout.addWidget(self.class_combo)

        self.train_button = QPushButton("Обучить YOLO")
        self.train_button.clicked.connect(self._train_yolo)
        control_layout.addWidget(self.train_button)

        self.annotate_button = QPushButton("Режим аннотации")
        self.annotate_button.setEnabled(False)
        self.annotate_button.clicked.connect(self._toggle_annotation)
        control_layout.addWidget(self.annotate_button)

        main_layout.addWidget(control_widget, 1)

        # Правая панель
        self.frame_viewer = FrameViewer()
        self.frame_viewer.update_status.connect(self.status_label.setText)
        main_layout.addWidget(self.frame_viewer, 3)

        # Стили
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
        if config_device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(config_device)

    def _update_device(self, device: str):
        self.config.update("device", device)
        self.device = self._get_device()
        if self.cnn_model:
            self.cnn_model.to(self.device)
        logging.info(f"Device updated: {self.device}")

    def _update_excavator(self, excavator: str):
        self.config.update("excavator", excavator)
        self.config.update("bucket_volume", self.excavators[excavator])
        logging.info(f"Excavator: {excavator}, Bucket volume: {self.excavators[excavator]} m³")

    def _update_class(self, class_name: str):
        self.frame_viewer.class_id = {"bucket": 0}.get(class_name, 0)
        logging.info(f"Annotation class: {class_name}")

    def _load_video(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Выберите видео", "", "Video Files (*.mp4 *.avi)")
        if file_name:
            self.status_label.setText(f"Загружено: {file_name}")
            self.progress_bar.setValue(0)
            self.frame_viewer.update_frame("", [])
            logging.info(f"Video: {file_name}")
            self._process_video(file_name)

    def _train_yolo(self):
        data_path, _ = QFileDialog.getOpenFileName(self, "Выберите data.yaml", "", "YAML Files (*.yaml)")
        if data_path:
            from developer.trainer import Trainer
            self.status_label.setText("Обучение YOLO...")
            self.train_button.setEnabled(False)
            trainer = Trainer(self.config)
            try:
                trainer.train_yolo(data_path, epochs=10)
                self.status_label.setText("Обучение завершено!")
            except Exception as e:
                self.status_label.setText(f"Ошибка обучения: {str(e)}")
            self.train_button.setEnabled(True)

    def _extract_frames(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Выберите видео", "", "Video Files (*.mp4 *.avi)")
        if file_name:
            self.status_label.setText(f"Извлечение кадров: {file_name}")
            self.processor.extract_frames(file_name)

    def _process_video(self, video_path: str):
        if not self.yolo_model:
            self.status_label.setText("Ошибка: YOLO не загружен")
            return
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        output_dir = f"data/{timestamp}"
        self.frame_viewer.no_bucket_frames = []
        self.frame_viewer.low_conf_frames = []
        self.frame_viewer.current_frame_index = -1
        self.processor = DeveloperProcessor(video_path, self.yolo_model, self.cnn_model, self.config, output_dir)
        self.processor.progress.connect(self.progress_bar.setValue)
        self.processor.status.connect(self.status_label.setText)
        self.processor.frame_processed.connect(self.frame_viewer.update_frame)
        self.processor.no_bucket_frame.connect(self.frame_viewer.add_no_bucket_frame)
        self.processor.low_conf_frame.connect(self.frame_viewer.add_low_conf_frame)
        self.processor.finished.connect(self._on_processing_finished)
        self.load_button.setEnabled(False)
        self.annotate_button.setEnabled(True)
        self.config.update("last_project", output_dir)
        self.processor.start()

    def _toggle_annotation(self):
        self.frame_viewer.review_mode = False
        self.frame_viewer.annotation_mode = not self.frame_viewer.annotation_mode
        status = "включён" if self.frame_viewer.annotation_mode else "выключён"
        self.status_label.setText(f"Режим аннотации: {status}")
        if self.frame_viewer.annotation_mode and self.frame_viewer.no_bucket_frames:
            self.frame_viewer.show_frame(0, True)
            self.frame_viewer.annotate_frame()
        elif self.frame_viewer.annotation_mode:
            self.status_label.setText("Режим аннотации: нет кадров без ковша")

    def _on_processing_finished(self):
        self.load_button.setEnabled(True)

    def start(self):
        self.show()