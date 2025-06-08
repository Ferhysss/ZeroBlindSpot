from typing import Optional
from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QLabel, QComboBox, QProgressBar, QTabWidget, QMenu, QAction
from PyQt5.QtCore import Qt
from core.interfaces.module import ModuleInterface
from core.config import Config
from developer.ui.frame_viewer import FrameViewer
from developer.processor import DeveloperProcessor
from developer.trainer import Trainer
from core.models.yolo_model import YoloModel
from core.models.cnn import SimpleCNN
import torch
import logging
import os
import yaml
from glob import glob

class DeveloperModule(QMainWindow, ModuleInterface):
    def __init__(self, project_dir: str, video_path: str):
        super().__init__()
        self.setWindowTitle("ZeroBlindSpot - Developer")
        self.setMinimumSize(1000, 600)
        self.config = Config()
        self.yolo_model: Optional[YoloModel] = None
        self.cnn_model: Optional[SimpleCNN] = None
        self.device = self._get_device()
        self.project_dir = project_dir
        self.video_path = video_path
        self.excavators = {
            "Экскаватор A": 1.5,
            "Экскаватор B": 2.0,
            "Экскаватор C": 2.5
        }
        self._init_models()
        self._init_ui()
        self._load_project()

    def _init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Панель управления
        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)

        # Вкладки
        self.tab_widget = QTabWidget()
        control_layout.addWidget(self.tab_widget)

        # Вкладка YOLO
        yolo_widget = QWidget()
        yolo_layout = QVBoxLayout(yolo_widget)
        self.status_label = QLabel("Загружен проект")
        yolo_layout.addWidget(self.status_label)
        self.progress_bar = QProgressBar()
        yolo_layout.addWidget(self.progress_bar)
        self.class_combo = QComboBox()
        self.class_combo.addItems(["bucket"])
        self.class_combo.currentTextChanged.connect(self._update_class)
        yolo_layout.addWidget(self.class_combo)
        self.annotate_button = QPushButton("Режим аннотации")
        self.annotate_button.setEnabled(False)
        self.annotate_button.clicked.connect(self._toggle_annotation)
        yolo_layout.addWidget(self.annotate_button)
        self.review_button = QPushButton("Режим ревью")
        self.review_button.setEnabled(False)
        self.review_button.clicked.connect(self._toggle_review)
        yolo_layout.addWidget(self.review_button)
        self.train_button = QPushButton("Обучить YOLO")
        self.train_button.clicked.connect(self._train_yolo)
        yolo_layout.addWidget(self.train_button)
        self.tab_widget.addTab(yolo_widget, "YOLO")

        # Вкладка CNN (заглушка)
        cnn_widget = QWidget()
        cnn_layout = QVBoxLayout(cnn_widget)
        cnn_label = QLabel("CNN: В разработке")
        cnn_layout.addWidget(cnn_label)
        self.cnn_annotate_button = QPushButton("Режим аннотации CNN")
        self.cnn_annotate_button.setEnabled(False)
        cnn_layout.addWidget(self.cnn_annotate_button)
        self.cnn_review_button = QPushButton("Режим ревью CNN")
        self.cnn_review_button.setEnabled(False)
        cnn_layout.addWidget(self.cnn_review_button)
        self.tab_widget.addTab(cnn_widget, "CNN")

        self.cnn_annotate_button.clicked.connect(self._toggle_cnn_annotation)
        self.cnn_review_button.clicked.connect(self._toggle_cnn_review)

        self.cnn_train_button = QPushButton("Обучить CNN")
        self.cnn_train_button.clicked.connect(self._train_cnn)
        cnn_layout.addWidget(self.cnn_train_button)

        self.cnn_class_combo = QComboBox()
        self.cnn_class_combo.addItems(["Нейтральный", "Зачерпывание", "Высыпание"])
        self.cnn_class_combo.currentTextChanged.connect(self._update_cnn_class)
        cnn_layout.addWidget(self.cnn_class_combo)

        # Вкладка Результаты
        results_widget = QWidget()
        results_layout = QVBoxLayout(results_widget)
        self.results_label = QLabel("Результаты: нет данных")
        results_layout.addWidget(self.results_label)
        self.tab_widget.addTab(results_widget, "Результаты")

        settings_widget = QWidget()
        settings_layout = QVBoxLayout(settings_widget)
        self.excavator_combo = QComboBox()
        self.excavator_combo.addItems(["Экскаватор A (1.5 м³)", "Экскаватор B (2.0 м³)", "Экскаватор C (2.5 м³)"])
        self.excavator_combo.currentTextChanged.connect(self._update_excavator)
        settings_layout.addWidget(QLabel("Экскаватор:"))
        settings_layout.addWidget(self.excavator_combo)
        self.tab_widget.addTab(settings_widget, "Настройки")

        # Вкладка Настройки
        settings_widget = QWidget()
        settings_layout = QVBoxLayout(settings_widget)
        self.device_combo = QComboBox()
        self.device_combo.addItems(["auto", "cpu"])
        if torch.cuda.is_available():
            self.device_combo.addItem("cuda")
        self.device_combo.currentTextChanged.connect(self._update_device)
        settings_layout.addWidget(self.device_combo)
        self.excavator_combo = QComboBox()
        self.excavator_combo.addItems(self.excavators.keys())
        self.excavator_combo.currentTextChanged.connect(self._update_excavator)
        excavator = self.config.get("excavator", "Экскаватор A")
        self.excavator_combo.setCurrentText(excavator)
        settings_layout.addWidget(self.excavator_combo)
        self.tab_widget.addTab(settings_widget, "Настройки")

        main_layout.addWidget(control_widget, 1)
        self.frame_viewer = FrameViewer()
        self.frame_viewer.update_status.connect(self.status_label.setText)
        main_layout.addWidget(self.frame_viewer, 3)

        try:
            with open("config/styles.qss", "r", encoding='utf-8') as f:
                self.setStyleSheet(f.read())
        except FileNotFoundError:
            logging.warning("styles.qss not found")

        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)

        

    def _show_context_menu(self, pos):
        menu = QMenu(self)
        yolo_action = QAction("Перейти к YOLO", self)
        yolo_action.triggered.connect(lambda: self.tab_widget.setCurrentIndex(0))
        menu.addAction(yolo_action)
        cnn_action = QAction("Перейти к CNN", self)
        cnn_action.triggered.connect(lambda: self.tab_widget.setCurrentIndex(1))
        menu.addAction(cnn_action)
        settings_action = QAction("Перейти к Настройкам", self)
        settings_action.triggered.connect(lambda: self.tab_widget.setCurrentIndex(2))
        menu.addAction(settings_action)
        extract_action = QAction("Извлечь кадры (заморожено)", self)
        extract_action.triggered.connect(self._extract_frames)
        menu.addAction(extract_action)
        menu.exec_(self.mapToGlobal(pos))

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
        volumes = {"Экскаватор A (1.5 м³)": 1.5, "Экскаватор B (2.0 м³)": 2.0, "Экскаватор C (2.5 м³)": 2.5}
        self.config["excavator"] = excavator.split(" ")[0]
        self.config["bucket_volume"] = volumes[excavator]
        logging.info(f"Selected excavator: {excavator}, volume: {self.config['bucket_volume']} m³")

    def _update_class(self, class_name: str):
        self.frame_viewer.class_id = {"bucket": 0}.get(class_name, 0)
        logging.info(f"Annotation class: {class_name}")

    def _load_project(self):
        if not os.path.exists(self.project_dir):
            self.status_label.setText("Ошибка: проект не найден")
            return
        project_config_path = os.path.join(self.project_dir, "project.yaml")
        if os.path.exists(project_config_path):
            with open(project_config_path, "r", encoding='utf-8') as f:
                project_config = yaml.safe_load(f)
                self.config.update("excavator", project_config.get("excavator", "Экскаватор A"))
                self.config.update("frame_rate", project_config.get("frame_rate", 1))
                self.excavator_combo.setCurrentText(project_config.get("excavator", "Экскаватор A"))
            self.status_label.setText(f"Проект загружен: {self.project_dir}")
            logging.info(f"Project loaded: {self.project_dir}")

            frames_dir = os.path.join(self.project_dir, "frames")
            no_bucket_dir = os.path.join(self.project_dir, "no_bucket")
            annotations_dir = os.path.join(self.project_dir, "annotations")
            negative_dir = os.path.join(self.project_dir, "negative")

            self.frame_viewer.no_bucket_frames = []
            self.frame_viewer.low_conf_frames = []

            if os.path.exists(frames_dir):
                annotated_frames = set()
                if os.path.exists(annotations_dir):
                    for ann_file in sorted(glob(os.path.join(annotations_dir, "*.txt"))):
                        frame_name = os.path.basename(ann_file).replace(".txt", ".jpg")
                        frame_path = os.path.join(frames_dir, frame_name)
                        if os.path.exists(frame_path):
                            annotations = []
                            try:
                                with open(ann_file, "r", encoding='utf-8') as f:
                                    for line in f:
                                        parts = line.strip().split()
                                        if len(parts) == 5:
                                            class_id, x_norm, y_norm, w_norm, h_norm = map(float, parts)
                                            img = cv2.imread(frame_path)
                                            if img is None:
                                                continue
                                            img_h, img_w = img.shape[:2]
                                            x = x_norm * img_w
                                            y = y_norm * img_h
                                            w = w_norm * img_w
                                            h = h_norm * img_h
                                            annotations.append((x, y, w, h, 1.0))
                                if annotations:
                                    self.frame_viewer.low_conf_frames.append((frame_path, annotations))
                                    self.frame_viewer.frame_annotations[frame_path] = annotations
                                    annotated_frames.add(frame_name)
                            except Exception as e:
                                logging.error(f"Failed to load annotation {ann_file}: {str(e)}")

                if os.path.exists(no_bucket_dir):
                    for frame_path in sorted(glob(os.path.join(no_bucket_dir, "*.jpg"))):
                        frame_name = os.path.basename(frame_path)
                        if frame_name not in annotated_frames:
                            self.frame_viewer.no_bucket_frames.append((frame_path, []))
                        else:
                            logging.info(f"Skipped annotated frame {frame_name} in no_bucket_dir")

                if os.path.exists(negative_dir):
                    negative_frames = glob(os.path.join(negative_dir, "*.jpg"))
                    for frame_path in negative_frames:
                        self.frame_viewer.frame_annotations[frame_path] = []

                if self.frame_viewer.no_bucket_frames or self.frame_viewer.low_conf_frames:
                    self.annotate_button.setEnabled(True)
                    self.review_button.setEnabled(True)
                    self.frame_viewer.update_counter()
                    logging.info(f"Loaded {len(self.frame_viewer.no_bucket_frames)} no_bucket, {len(self.frame_viewer.low_conf_frames)} low_conf frames")
                else:
                    self._process_video()
            else:
                self._process_video()
        else:
            self.status_label.setText("Ошибка: конфигурация проекта не найдена")

    def _save_project_config(self):
        project_config = {
            "video_path": self.video_path,
            "excavator": self.config.get("excavator", "Экскаватор A"),
            "frame_rate": self.config.get("frame_rate", 1),
            "project_dir": self.project_dir
        }
        with open(os.path.join(self.project_dir, "project.yaml"), "w", encoding='utf-8') as f:
            yaml.safe_dump(project_config, f)

    def _extract_frames(self):
        if not self.video_path:
            self.status_label.setText("Ошибка: видео не выбрано")
            logging.error("No video path specified for frame extraction")
            return
        if not hasattr(self, 'processor') or self.processor is None:
            if not self.yolo_model:
                self.status_label.setText("Ошибка: YOLO не загружен")
                return
            self.processor = DeveloperProcessor(self.video_path, self.yolo_model, self.cnn_model, self.config, self.project_dir)
            self.processor.progress.connect(self.progress_bar.setValue)
            self.processor.status.connect(self.status_label.setText)
            self.processor.frame_processed.connect(self.frame_viewer.update_frame)
            self.processor.no_bucket_frame.connect(self.frame_viewer.add_no_bucket_frame)
            self.processor.low_conf_frame.connect(self.frame_viewer.add_low_conf_frame)
            self.processor.finished.connect(self._on_processing_finished)
        self.status_label.setText(f"Извлечение кадров: {self.video_path}")
        self.processor.extract_frames(self.video_path)

    def _train_yolo(self):
        data_path, _ = QFileDialog.getOpenFileName(self, "Выберите data.yaml", "", "YAML Files (*.yaml)")
        if data_path:
            self.status_label.setText("Обучение YOLO...")
            self.train_button.setEnabled(False)
            trainer = Trainer(self.config)
            try:
                trainer.train_yolo(data_path, epochs=10)
                self.status_label.setText("Обучение завершено")
            except Exception as e:
                self.status_label.setText(f"Ошибка обучения: {str(e)}")
            self.train_button.setEnabled(True)

    def _process_video(self):
        if not self.yolo_model:
            self.status_label.setText("Ошибка: YOLO не загружен")
            return
        os.makedirs(self.project_dir, exist_ok=True)
        self.frame_viewer.no_bucket_frames = []
        self.frame_viewer.low_conf_frames = []
        self.frame_viewer.current_frame_index = -1
        self.processor = DeveloperProcessor(self.video_path, self.yolo_model, self.cnn_model, self.config, self.project_dir)
        self.processor.progress.connect(self.progress_bar.setValue)
        self.processor.status.connect(self.status_label.setText)
        self.processor.frame_processed.connect(self.frame_viewer.update_frame)
        self.processor.no_bucket_frame.connect(self.frame_viewer.add_no_bucket_frame)
        self.processor.low_conf_frame.connect(self.frame_viewer.add_low_conf_frame)
        self.processor.finished.connect(self._on_processing_finished)
        self.annotate_button.setEnabled(True)
        self.review_button.setEnabled(True)
        self.config.update("last_project", self.project_dir)
        self._save_project_config()
        self.processor.start()

    def _toggle_annotation(self):
        self.frame_viewer.review_mode = False
        self.frame_viewer.annotation_mode = not self.frame_viewer.annotation_mode
        status = "включён" if self.frame_viewer.annotation_mode else "выключён"
        self.status_label.setText(f"Режим аннотации: {status}")
        if self.frame_viewer.annotation_mode and self.frame_viewer.no_bucket_frames:
            self.frame_viewer.show_frame(0, False)
            self.frame_viewer.annotate_frame()
        elif self.frame_viewer.annotation_mode:
            self.status_label.setText("Режим аннотации: нет кадров без ковша")
        self.frame_viewer.update_counter()

    def _toggle_review(self, checked):
        if checked and not self.frame_viewer.low_conf_frames:
            self.status_label.setText("Нет низкоконфиденциальных кадров")
            return
        self.frame_viewer.review_mode = checked
        self.frame_viewer.show_frame(0, checked)
        self.frame_viewer.annotation_mode = False
        self.frame_viewer.review_mode = not self.frame_viewer.review_mode
        status = "включён" if self.frame_viewer.review_mode else "выключён"
        self.status_label.setText(f"Режим ревью: {status}")
        if self.frame_viewer.review_mode and self.frame_viewer.low_conf_frames:
            self.frame_viewer.show_frame(0, True)
            self.frame_viewer.annotate_frame()
        elif self.frame_viewer.review_mode:
            self.status_label.setText("Режим ревью: нет низкоконфиденциальных кадров")
        self.frame_viewer.current_frame_path = ""
        self.frame_viewer.current_frame_index = -1
        self.frame_viewer.update_counter()

    def _toggle_cnn_annotation(self):
        self.frame_viewer.review_mode = False
        self.frame_viewer.annotation_mode = False
        self.frame_viewer.cnn_review_mode = False
        self.frame_viewer.cnn_annotation_mode = not self.frame_viewer.cnn_annotation_mode
        status = "включён" if self.frame_viewer.cnn_annotation_mode else "выключён"
        self.status_label.setText(f"Режим аннотации CNN: {status}")
        if self.frame_viewer.cnn_annotation_mode and self.frame_viewer.no_bucket_frames:
            self.frame_viewer.show_next_frame()
            self.frame_viewer.annotate_frame()
        elif self.frame_viewer.cnn_annotation_mode:
            self.status_label.setText("Режим аннотации CNN: нет кадров")
        self.frame_viewer.update_counter()

    def _toggle_cnn_review(self):
        self.frame_viewer.annotation_mode = False
        self.frame_viewer.review_mode = False
        self.frame_viewer.cnn_annotation_mode = False
        self.frame_viewer.cnn_review_mode = not self.frame_viewer.cnn_review_mode
        status = "включён" if self.frame_viewer.cnn_review_mode else "выключён"
        self.status_label.setText(f"Режим ревью CNN: {status}")
        if self.frame_viewer.cnn_review_mode and self.frame_viewer.cnn_annotations:
            self.frame_viewer.show_next_frame()
            self.frame_viewer.annotate_frame()
        elif self.frame_viewer.cnn_review_mode:
            self.status_label.setText("Режим ревью CNN: нет аннотаций")
        self.frame_viewer.current_frame_path = ""
        self.frame_viewer.current_frame_index = -1
        self.frame_viewer.update_counter()

    def _train_cnn(self):
        data_dir = QFileDialog.getExistingDirectory(self, "Выберите папку с CNN аннотациями")
        if data_dir:
            self.status_label.setText("Обучение CNN...")
            self.cnn_annotate_button.setEnabled(False)
            try:
                # Заглушка для обучения CNN
                logging.info(f"Training CNN with data from {data_dir}")
                self.status_label.setText("Обучение CNN завершено")
            except Exception as e:
                self.status_label.setText(f"Ошибка обучения CNN: {str(e)}")
            self.cnn_annotate_button.setEnabled(True)

    def _update_cnn_class(self, class_name: str):
        self.frame_viewer.class_id = {"Нейтральный": 0, "Зачерпывание": 1, "Высыпание": 2}.get(class_name, 0)
        logging.info(f"CNN annotation class: {class_name}")

    def _init_processor(self):
        yolo_model = YoloModel("models/yolo.pt")
        cnn_model = SimpleCNN().to("cuda" if torch.cuda.is_available() else "cpu")
        try:
            cnn_model.load_state_dict(torch.load("models/bucket_cnn.pth", map_location=cnn_model.device))
            cnn_model.eval()
        except FileNotFoundError:
            logging.error("CNN model file not found: models/cnn.pth")
            self.status_label.setText("Ошибка: CNN модель не найдена")
            return
        self.processor = Processor(self.video_path, yolo_model, cnn_model, self.config, self.project_dir)
        self.processor.progress.connect(self._update_progress)
        self.processor.status.connect(self.status_label.setText)
        self.processor.frame_processed.connect(self.frame_viewer.update_frame)
        self.processor.no_bucket_frame.connect(self.frame_viewer.add_no_bucket_frame)
        self.processor.low_conf_frame.connect(self.frame_viewer.add_low_conf_frame)
        self.processor.finished.connect(self._on_processing_finished)

    

    def _on_processing_finished(self):
        self.status_label.setText("Обработка видео завершена")
        logging.info("Video processing finished")
        result_path = os.path.join(self.project_dir, "results", "summary.txt")
        if os.path.exists(result_path):
            with open(result_path, "r", encoding='utf-8') as f:
                self.results_label.setText(f.read().replace("\n", "<br>"))

    def start(self):
        self.show()