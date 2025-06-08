from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QScrollArea
from PyQt5.QtCore import Qt, pyqtSignal, QPoint
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen
import cv2
import os
import logging
from typing import List, Tuple, Optional, Dict

class FrameViewer(QWidget):
    update_status = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMouseTracking(True)
        self.current_frame_path: str = ""
        self.current_annotations: List[Tuple[float, float, float, float, float, int]] = []
        self.no_bucket_frames: List[Tuple[str, List[Tuple[float, float, float, float, float, int]]]] = []
        self.low_conf_frames: List[Tuple[str, List[Tuple[float, float, float, float, float, int]]]] = []
        self.current_frame_index: int = -1
        self.annotation_mode: bool = False
        self.review_mode: bool = False
        self.cnn_annotation_mode: bool = False
        self.cnn_review_mode: bool = False
        self.start_point: Optional[QPoint] = None
        self.end_point: Optional[QPoint] = None
        self.drawing: bool = False
        self.class_id: int = 0
        self.image_scale: float = 1.0
        self.image_size: Optional[Tuple[int, int]] = None
        self.frame_annotations: Dict[str, List[Tuple[float, float, float, float, float, int]]] = {}
        self.cnn_annotations: Dict[str, int] = {}  # Для CNN: {frame_path: class_id}
        self.scale_cache: Dict[str, float] = {}

        main_layout = QVBoxLayout()
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.image_label)
        scroll_area.setWidgetResizable(True)
        main_layout.addWidget(scroll_area, stretch=1)

        self.counter_label = QLabel("Кадры: 0/0")
        main_layout.addWidget(self.counter_label)

        button_widget = QWidget()
        button_layout = QVBoxLayout(button_widget)
        button_widget.setMinimumHeight(150)

        self.save_button = QPushButton("Сохранить аннотации")
        self.save_button.clicked.connect(self.save_annotations)
        self.save_button.setEnabled(False)
        button_layout.addWidget(self.save_button)

        self.no_bucket_button = QPushButton("Отсутствует ковш")
        self.no_bucket_button.clicked.connect(self.mark_no_bucket)
        self.no_bucket_button.setEnabled(False)
        button_layout.addWidget(self.no_bucket_button)

        self.next_button = QPushButton("Следующий кадр")
        self.next_button.clicked.connect(self.show_next_frame)
        self.next_button.setEnabled(False)
        button_layout.addWidget(self.next_button)

        self.prev_button = QPushButton("Предыдущий кадр")
        self.prev_button.clicked.connect(self.show_prev_frame)
        self.prev_button.setEnabled(False)
        button_layout.addWidget(self.prev_button)

        self.confirm_button = QPushButton("Подтвердить детекцию")
        self.confirm_button.clicked.connect(self.confirm_detection)
        self.confirm_button.setEnabled(False)
        button_layout.addWidget(self.confirm_button)

        self.delete_button = QPushButton("Удалить детекцию")
        self.delete_button.clicked.connect(self.delete_detection)
        self.delete_button.setEnabled(False)
        button_layout.addWidget(self.delete_button)

        main_layout.addWidget(button_widget)
        self.setLayout(main_layout)

        self.image_label.mousePressEvent = self.mouse_press
        self.image_label.mouseMoveEvent = self.mouse_move
        self.image_label.mouseReleaseEvent = self.mouse_release
        self.image_label.setFocusPolicy(Qt.StrongFocus)
        self.setFocusPolicy(Qt.StrongFocus)

    def update_frame(self, frame_path: str, annotations: List[Tuple[float, float, float, float, float, int]]):
        self.current_frame_path = frame_path
        self.current_annotations = annotations
        self.frame_annotations[frame_path] = annotations
        if self.annotation_mode or self.review_mode or self.cnn_annotation_mode or self.cnn_review_mode:
            self.display_frame()
        self.update_counter()

    def add_no_bucket_frame(self, frame_path: str):
        self.no_bucket_frames.append((frame_path, []))
        self.update_counter()

    def add_low_conf_frame(self, frame_path: str, annotations: List[Tuple[float, float, float, float, float, int]]):
        self.low_conf_frames.append((frame_path, annotations))
        self.update_counter()

    def update_counter(self):
        if self.annotation_mode:
            total = len(self.no_bucket_frames)
            current = self.current_frame_index + 1 if total > 0 else 0
            annotated_count = sum(1 for _, anns in self.no_bucket_frames if anns)
            self.counter_label.setText(f"Кадры для аннотации: {current}/{total} (нарисовано: {annotated_count})")
            if total == 0:
                self.update_status.emit("Режим аннотации: нет кадров")
            elif annotated_count == total:
                self.update_status.emit("Все кадры аннотированы")
        elif self.review_mode:
            total = len(self.low_conf_frames)
            current = self.current_frame_index + 1 if total > 0 else 0
            self.counter_label.setText(f"Кадры для ревью: {current}/{total}")
            if total == 0:
                self.update_status.emit("Режим ревью: нет низкоконфиденциальных кадров")
        elif self.cnn_annotation_mode:
            total = len(self.no_bucket_frames)
            current = self.current_frame_index + 1 if total > 0 else 0
            annotated_count = sum(1 for path in self.cnn_annotations if path in [p for p, _ in self.no_bucket_frames])
            self.counter_label.setText(f"CNN аннотации: {current}/{total} (нарисовано: {annotated_count})")
        elif self.cnn_review_mode:
            total = len(self.cnn_annotations)
            current = self.current_frame_index + 1 if total > 0 else 0
            self.counter_label.setText(f"CNN ревью: {current}/{total}")
        else:
            self.counter_label.setText("Кадры: 0/0")

    def display_frame(self):
        if not os.path.exists(self.current_frame_path):
            self.image_label.clear()
            return
        frame = cv2.imread(self.current_frame_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]

        if self.current_frame_path in self.scale_cache:
            self.image_scale = self.scale_cache[self.current_frame_path]
            self.image_size = (w, h)
        else:
            self.image_size = (w, h)
            available_height = self.height() - 170
            self.image_scale = min(self.width() / w, available_height / h) * 0.9
            self.scale_cache[self.current_frame_path] = self.image_scale
            logging.info(f"Image scale set to {self.image_scale} for {self.current_frame_path}")

        new_w, new_h = int(w * self.image_scale), int(h * self.image_scale)
        frame = cv2.resize(frame, (new_w, new_h))

        self.current_annotations = self.frame_annotations.get(self.current_frame_path, [])

        if self.annotation_mode or self.review_mode:
            for x, y, w, h, conf, class_id in self.current_annotations:
                x1 = int((x - w / 2) * self.image_scale)
                y1 = int((y - h / 2) * self.image_scale)
                x2 = int((x + w / 2) * self.image_scale)
                y2 = int((y + h / 2) * self.image_scale)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"conf: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        elif self.cnn_annotation_mode or self.cnn_review_mode:
            if self.current_frame_path in self.cnn_annotations:
                class_id = self.cnn_annotations[self.current_frame_path]
                cv2.putText(frame, f"CNN class: {class_id}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        if self.drawing and self.start_point and self.end_point:
            x1, y1 = self.start_point.x(), self.start_point.y()
            x2, y2 = self.end_point.x(), self.end_point.y()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        q_image = QImage(frame.data, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(q_image))
        self.image_label.setFixedSize(new_w, new_h)

    def mouse_press(self, event):
        if (self.annotation_mode or self.cnn_annotation_mode) and event.button() == Qt.LeftButton:
            self.start_point = event.pos()
            self.drawing = True
            self.image_label.setFocus()
            logging.info(f"Mouse press at {self.start_point} (scale: {self.image_scale}, size: {self.image_size})")

    def mouse_move(self, event):
        if (self.annotation_mode or self.cnn_annotation_mode) and self.drawing:
            self.end_point = event.pos()
            self.display_frame()
            logging.debug(f"Mouse move to {self.end_point}")

    def mouse_release(self, event):
        if (self.annotation_mode or self.cnn_annotation_mode) and event.button() == Qt.LeftButton:
            self.end_point = event.pos()
            self.drawing = False
            if self.annotation_mode:
                self.add_annotation()
            elif self.cnn_annotation_mode:
                self.add_cnn_annotation()
            self.display_frame()
            logging.info(f"Mouse release at {self.end_point}")
            if self.start_point == self.end_point:
                logging.debug("Click without drag ignored")
            self.start_point = None
            self.end_point = None

    def add_annotation(self):
        if self.start_point and self.end_point and self.image_size:
            img_w, img_h = self.image_size
            x1, y1 = self.start_point.x() / self.image_scale, self.start_point.y() / self.image_scale
            x2, y2 = self.end_point.x() / self.image_scale, self.end_point.y() / self.image_scale
            if x1 == x2 or y1 == y2:
                logging.warning("Zero-sized annotation skipped")
                return
            x = (x1 + x2) / 2
            y = (y1 + y2) / 2
            w = abs(x2 - x1)
            h = abs(y2 - y1)
            self.current_annotations.append((x, y, w, h, 1.0, self.class_id))
            self.frame_annotations[self.current_frame_path] = self.current_annotations
            self.no_bucket_frames[self.current_frame_index] = (self.current_frame_path, self.current_annotations)
            self.save_button.setEnabled(True)
            logging.info(f"Annotation added: x={x}, y={y}, w={w}, h={h}, class_id={self.class_id}")
            self.update_counter()

    def add_cnn_annotation(self):
        if self.current_frame_path:
            self.cnn_annotations[self.current_frame_path] = self.class_id
            self.save_button.setEnabled(True)
            logging.info(f"CNN annotation added: {self.current_frame_path}, class={self.class_id}")
            self.update_counter()

    def save_annotations(self):
        if self.annotation_mode:
            annotated_frames = [(path, anns) for path, anns in self.no_bucket_frames if anns]
            if not annotated_frames:
                logging.warning("No annotated frames to save")
                self.update_status.emit("Нет нарисованных аннотаций")
                return
            output_dir = os.path.dirname(os.path.dirname(annotated_frames[0][0]))
            annotation_dir = f"{output_dir}/annotations"
            frames_dir = f"{output_dir}/frames"
            dataset_dir = os.path.join(os.path.dirname(output_dir), "datasets", "train")
            dataset_images_dir = os.path.join(dataset_dir, "images")
            dataset_labels_dir = os.path.join(dataset_dir, "labels")
            os.makedirs(annotation_dir, exist_ok=True)
            os.makedirs(frames_dir, exist_ok=True)
            os.makedirs(dataset_images_dir, exist_ok=True)
            os.makedirs(dataset_labels_dir, exist_ok=True)
            project_id = os.path.basename(output_dir).replace(" ", "_")
            saved_count = 0
            for frame_path, annotations in annotated_frames:
                frame_file = os.path.basename(frame_path)
                annotation_path = f"{annotation_dir}/{frame_file}.txt"
                new_frame_path = f"{frames_dir}/{frame_file}"
                dataset_image_path = f"{dataset_images_dir}/{project_id}_{frame_file}"
                dataset_label_path = f"{dataset_labels_dir}/{project_id}_{frame_file}.txt"
                try:
                    img = cv2.imread(frame_path)
                    if img is None:
                        logging.warning(f"Failed to load image {frame_path}")
                        continue
                    img_h, img_w = img.shape[:2]
                    with open(annotation_path, "w", encoding='utf-8') as f:
                        for x, y, w, h, conf, class_id in annotations:
                            x_norm = x / img_w
                            y_norm = y / img_h
                            w_norm = w / img_w
                            h_norm = h / img_h
                            f.write(f"{class_id} {x_norm:.6f} {y_norm:.6f} {w_norm:.6f} {h_norm:.6f}\n")
                    with open(dataset_label_path, "w", encoding='utf-8') as f:
                        for x, y, w, h, conf, class_id in annotations:
                            x_norm = x / img_w
                            y_norm = y / img_h
                            w_norm = w / img_w
                            h_norm = h / img_h
                            f.write(f"{class_id} {x_norm:.6f} {y_norm:.6f} {w_norm:.6f} {h_norm:.6f}\n")
                    cv2.imwrite(dataset_image_path, img)
                    if os.path.exists(new_frame_path):
                        os.remove(new_frame_path)
                    os.rename(frame_path, new_frame_path)
                    logging.info(f"Annotations saved for {frame_file} at {annotation_path}")
                    logging.info(f"Frame moved to {new_frame_path}")
                    logging.info(f"Dataset updated: {dataset_image_path}, {dataset_label_path}")
                    saved_count += 1
                    self.low_conf_frames.append((new_frame_path, annotations))
                except Exception as e:
                    logging.error(f"Failed to save annotations for {frame_file}: {str(e)}")
                    self.update_status.emit(f"Ошибка сохранения: {frame_file}")
            self.no_bucket_frames = [(path, anns) for path, anns in self.no_bucket_frames if not anns]
            self.current_frame_index = -1
            self.show_next_frame()
            self.update_status.emit(f"Сохранено аннотаций: {saved_count}")
            self.save_button.setEnabled(False)
            self.update_counter()
        elif self.cnn_annotation_mode:
            output_dir = os.path.dirname(os.path.dirname(self.current_frame_path))
            cnn_annotation_dir = f"{output_dir}/cnn_annotations"
            os.makedirs(cnn_annotation_dir, exist_ok=True)
            saved_count = 0
            for frame_path, class_id in self.cnn_annotations.items():
                frame_file = os.path.basename(frame_path)
                annotation_path = f"{cnn_annotation_dir}/{frame_file}.txt"
                try:
                    with open(annotation_path, "w", encoding='utf-8') as f:
                        f.write(f"{class_id}\n")
                    logging.info(f"CNN annotation saved for {frame_file} at {annotation_path}")
                    saved_count += 1
                except Exception as e:
                    logging.error(f"Failed to save CNN annotation for {frame_file}: {str(e)}")
                    self.update_status.emit(f"Ошибка сохранения: {frame_file}")
            self.update_status.emit(f"Сохранено CNN аннотаций: {saved_count}")
            self.save_button.setEnabled(False)
            self.update_counter()

    def mark_no_bucket(self):
        if not self.current_frame_path:
            logging.warning("No frame to mark as no bucket")
            return
        frame_file = os.path.basename(self.current_frame_path)
        output_dir = os.path.dirname(os.path.dirname(self.current_frame_path))
        negative_dir = f"{output_dir}/negative"
        os.makedirs(negative_dir, exist_ok=True)
        negative_path = f"{negative_dir}/{frame_file}"
        try:
            if os.path.exists(negative_path):
                os.remove(negative_path)
            os.rename(self.current_frame_path, negative_path)
            self.current_annotations = []
            self.frame_annotations[self.current_frame_path] = []
            self.no_bucket_frames.pop(self.current_frame_index)
            self.current_frame_index = max(0, self.current_frame_index - 1)
            logging.info(f"Frame marked as no bucket: {frame_file} at {negative_path}")
            self.update_status.emit(f"Ковш отсутствует: {frame_file}")
            self.show_next_frame()
            self.update_counter()
        except Exception as e:
            logging.error(f"Failed to mark no bucket: {str(e)}")
            self.update_status.emit(f"Ошибка: {str(e)}")

    def confirm_detection(self):
        if not self.current_frame_path or not self.current_annotations:
            logging.warning("No frame or annotations to confirm")
            return
        self.low_conf_frames = [(path, anns) for path, anns in self.low_conf_frames if path != self.current_frame_path]
        self.show_next_frame()
        self.update_counter()

    def delete_detection(self):
        if not self.current_frame_path:
            logging.warning("No frame to delete")
            return
        frame_file = os.path.basename(self.current_frame_path)
        output_dir = os.path.dirname(os.path.dirname(self.current_frame_path))
        negative_dir = f"{output_dir}/negative"
        os.makedirs(negative_dir, exist_ok=True)
        negative_path = f"{negative_dir}/{frame_file}"
        try:
            if os.path.exists(negative_path):
                os.remove(negative_path)
            os.rename(self.current_frame_path, negative_path)
            self.current_annotations = []
            self.frame_annotations[self.current_frame_path] = []
            self.low_conf_frames = [(path, anns) for path, anns in self.low_conf_frames if path != self.current_frame_path]
            logging.info(f"Detection deleted for {frame_file} at {negative_path}")
            self.update_status.emit(f"Детекция удалена: {frame_file}")
            self.show_next_frame()
            self.update_counter()
        except Exception as e:
            logging.error(f"Failed to delete detection: {str(e)}")
            self.update_status.emit(f"Ошибка удаления: {str(e)}")

    def annotate_frame(self):
        annotated_count = sum(1 for _, anns in self.no_bucket_frames if anns)
        self.save_button.setEnabled((self.annotation_mode and annotated_count > 0) or (self.cnn_annotation_mode and bool(self.cnn_annotations)))
        self.no_bucket_button.setEnabled(self.annotation_mode or self.cnn_annotation_mode)
        self.confirm_button.setEnabled(self.review_mode and bool(self.current_annotations))
        self.delete_button.setEnabled(self.review_mode)
        self.next_button.setEnabled(self.annotation_mode or (self.review_mode and bool(self.low_conf_frames)) or self.cnn_annotation_mode or self.cnn_review_mode)
        self.prev_button.setEnabled(self.annotation_mode or (self.review_mode and bool(self.low_conf_frames)) or self.cnn_annotation_mode or self.cnn_review_mode)
        self.display_frame()
        self.update_counter()

    def show_next_frame(self):
        if self.annotation_mode:
            if not self.no_bucket_frames:
                self.update_status.emit("Режим аннотации: нет кадров")
                self.current_frame_path = ""
                self.display_frame()
                self.update_counter()
                return
            self.current_frame_index = min(self.current_frame_index + 1, len(self.no_bucket_frames) - 1)
            self.current_frame_path, self.current_annotations = self.no_bucket_frames[self.current_frame_index]
            self.frame_annotations[self.current_frame_path] = self.current_annotations
            self.display_frame()
            self.update_counter()
        elif self.review_mode and self.low_conf_frames:
            self.current_frame_index = min(self.current_frame_index + 1, len(self.low_conf_frames) - 1)
            if self.current_frame_index >= 0:
                self.current_frame_path, self.current_annotations = self.low_conf_frames[self.current_frame_index]
                self.frame_annotations[self.current_frame_path] = self.current_annotations
                self.display_frame()
            else:
                self.update_status.emit("Режим ревью: нет низкоконфиденциальных кадров")
            self.update_counter()
        elif self.cnn_annotation_mode:
            total = len(self.no_bucket_frames)
            if total == 0:
                self.update_status.emit("Режим аннотации CNN: нет кадров")
                self.current_frame_path = ""
                self.display_frame()
                self.update_counter()
                return
            self.current_frame_index = min(self.current_frame_index + 1, total - 1)
            self.current_frame_path, _ = self.no_bucket_frames[self.current_frame_index]
            self.display_frame()
            self.update_counter()
        elif self.cnn_review_mode:
            cnn_frames = list(self.cnn_annotations.keys())
            if not cnn_frames:
                self.update_status.emit("Режим ревью CNN: нет аннотаций")
                self.current_frame_path = ""
                self.display_frame()
                self.update_counter()
                return
            self.current_frame_index = min(self.current_frame_index + 1, len(cnn_frames) - 1)
            self.current_frame_path = cnn_frames[self.current_frame_index]
            self.display_frame()
            self.update_counter()

    def show_prev_frame(self):
        if self.annotation_mode:
            total = len(self.no_bucket_frames)
            if total == 0:
                self.update_status.emit("Режим аннотации: нет кадров")
                self.current_frame_path = ""
                self.display_frame()
                self.update_counter()
                return
            self.current_frame_index = max(self.current_frame_index - 1, 0)
            self.current_frame_path, self.current_annotations = self.no_bucket_frames[self.current_frame_index]
            self.frame_annotations[self.current_frame_path] = self.current_annotations
            self.display_frame()
            self.update_counter()
        elif self.review_mode and self.low_conf_frames:
            self.current_frame_index = max(self.current_frame_index - 1, 0)
            if self.current_frame_index < len(self.low_conf_frames):
                self.current_frame_path, self.current_annotations = self.low_conf_frames[self.current_frame_index]
                self.frame_annotations[self.current_frame_path] = self.current_annotations
                self.display_frame()
            else:
                self.update_status.emit("Режим ревью: нет низкоконфиденциальных кадров")
            self.update_counter()
        elif self.cnn_annotation_mode:
            total = len(self.no_bucket_frames)
            if total == 0:
                self.update_status.emit("Режим аннотации CNN: нет кадров")
                self.current_frame_path = ""
                self.display_frame()
                self.update_counter()
                return
            self.current_frame_index = max(self.current_frame_index - 1, 0)
            self.current_frame_path, _ = self.no_bucket_frames[self.current_frame_index]
            self.display_frame()
            self.update_counter()
        elif self.cnn_review_mode:
            cnn_frames = list(self.cnn_annotations.keys())
            if not cnn_frames:
                self.update_status.emit("Режим ревью CNN: нет аннотаций")
                self.current_frame_path = ""
                self.display_frame()
                self.update_counter()
                return
            self.current_frame_index = max(self.current_frame_index - 1, 0)
            self.current_frame_path = cnn_frames[self.current_frame_index]
            self.display_frame()
            self.update_counter()

    def show_frame(self, index: int, review: bool):
        self.current_frame_index = index
        self.review_mode = review
        if review and self.low_conf_frames:
            if index < len(self.low_conf_frames):
                self.current_frame_path, self.current_annotations = self.low_conf_frames[index]
                self.frame_annotations[self.current_frame_path] = self.current_annotations
                self.display_frame()
            else:
                self.current_frame_path = ""
                self.update_status.emit("Режим ревью: нет низкоконфиденциальных кадров")
        elif self.no_bucket_frames:
            if index < len(self.no_bucket_frames):
                self.current_frame_path, self.current_annotations = self.no_bucket_frames[index]
                self.frame_annotations[self.current_frame_path] = self.current_annotations
                self.display_frame()
            else:
                self.current_frame_path = ""
                self.update_status.emit("Режим аннотации: нет кадров")
        else:
            self.current_frame_path = ""
            self.update_status.emit("Режим аннотации: нет кадров")
        self.update_counter()

    def keyPressEvent(self, event):
        if self.annotation_mode or self.review_mode or self.cnn_annotation_mode or self.cnn_review_mode:
            if event.key() == Qt.Key_D:
                self.show_next_frame()
            elif event.key() == Qt.Key_A:
                self.show_prev_frame()