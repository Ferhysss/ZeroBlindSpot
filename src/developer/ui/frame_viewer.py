from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QScrollArea
from PyQt5.QtCore import Qt, pyqtSignal, QPoint
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen
import cv2
import os
import logging
from typing import List, Tuple, Optional

class FrameViewer(QWidget):
    update_status = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.current_frame_path: str = ""
        self.current_annotations: List[Tuple[float, float, float, float, float]] = []
        self.no_bucket_frames: List[str] = []
        self.low_conf_frames: List[Tuple[str, List[Tuple[float, float, float, float, float]]]] = []
        self.current_frame_index: int = -1
        self.annotation_mode: bool = False
        self.review_mode: bool = False
        self.start_point: Optional[QPoint] = None
        self.end_point: Optional[QPoint] = None
        self.drawing: bool = False
        self.class_id: int = 0
        self.image_scale: float = 1.0
        self.image_size: Optional[Tuple[int, int]] = None

        main_layout = QVBoxLayout()
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.image_label)
        scroll_area.setWidgetResizable(True)
        main_layout.addWidget(scroll_area, stretch=1)

        button_widget = QWidget()
        button_layout = QVBoxLayout(button_widget)
        button_widget.setMinimumHeight(150)

        self.save_button = QPushButton("Сохранить аннотацию")
        self.save_button.clicked.connect(self.save_annotations)
        self.save_button.setEnabled(False)
        button_layout.addWidget(self.save_button)

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

    def update_frame(self, frame_path: str, annotations: List[Tuple[float, float, float, float, float]]):
        self.current_frame_path = frame_path
        self.current_annotations = annotations
        if self.annotation_mode or self.review_mode:
            self.display_frame()

    def add_no_bucket_frame(self, frame_path: str):
        self.no_bucket_frames.append(frame_path)

    def add_low_conf_frame(self, frame_path: str, annotations: List[Tuple[float, float, float, float, float]]):
        self.low_conf_frames.append((frame_path, annotations))

    def display_frame(self):
        if not os.path.exists(self.current_frame_path):
            return
        frame = cv2.imread(self.current_frame_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]

        if not self.image_size:
            self.image_size = (w, h)
            scale_w = (self.width() - 20) / w  # Учитываем отступы
            scale_h = (self.height() - 170) / h  # Учитываем кнопки
            self.image_scale = min(scale_w, scale_h) * 0.9  # Уменьшаем масштаб
            logging.info(f"Image scale set to {self.image_scale}")

        new_w, new_h = int(w * self.image_scale), int(h * self.image_scale)
        frame = cv2.resize(frame, (new_w, new_h))

        for x, y, w, h, conf in self.current_annotations:
            x1 = int((x - w / 2) * self.image_scale)
            y1 = int((y - h / 2) * self.image_scale)
            x2 = int((x + w / 2) * self.image_scale)
            y2 = int((y + h / 2) * self.image_scale)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"conf: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if self.drawing and self.start_point and self.end_point:
            x1, y1 = self.start_point.x(), self.start_point.y()
            x2, y2 = self.end_point.x(), self.end_point.y()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        q_image = QImage(frame.data, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(q_image))
        self.image_label.setFixedSize(new_w, new_h)  # Фиксируем размер

    def mouse_press(self, event):
        if self.annotation_mode and event.button() == Qt.LeftButton:
            self.start_point = event.pos()
            self.drawing = True
            logging.info(f"Mouse press at {self.start_point}")

    def mouse_move(self, event):
        if self.annotation_mode and self.drawing:
            self.end_point = event.pos()
            self.display_frame()
            logging.debug(f"Mouse move to {self.end_point}")

    def mouse_release(self, event):
        if self.annotation_mode and event.button() == Qt.LeftButton:
            self.end_point = event.pos()
            self.drawing = False
            self.add_annotation()
            self.display_frame()
            logging.info(f"Mouse release at {self.end_point}")
            self.start_point = None
            self.end_point = None

    def add_annotation(self):
        if self.start_point and self.end_point and self.image_size:
            img_w, img_h = self.image_size
            x1, y1 = self.start_point.x() / self.image_scale, self.start_point.y() / self.image_scale
            x2, y2 = self.end_point.x() / self.image_scale, self.end_point.y() / self.image_scale
            x = (x1 + x2) / 2
            y = (y1 + y2) / 2
            w = abs(x2 - x1)
            h = abs(y2 - y1)
            self.current_annotations.append((x, y, w, h, 1.0))
            self.save_button.setEnabled(True)
            logging.info(f"Annotation added: x={x}, y={y}, w={w}, h={h}")

    def save_annotations(self):
        if not self.current_frame_path or not self.current_annotations:
            logging.warning("No frame or annotations to save")
            return
        frame_file = os.path.basename(self.current_frame_path)
        output_dir = os.path.dirname(os.path.dirname(self.current_frame_path))
        annotation_dir = f"{output_dir}/annotations"
        os.makedirs(annotation_dir, exist_ok=True)
        annotation_path = f"{annotation_dir}/{frame_file}.txt"
        try:
            with open(annotation_path, "w", encoding='utf-8') as f:
                for x, y, w, h, conf in self.current_annotations:
                    img_w, img_h = self.image_size
                    x_norm = x / img_w
                    y_norm = y / img_h
                    w_norm = w / img_w
                    h_norm = h / img_h
                    f.write(f"{self.class_id} {x_norm:.6f} {y_norm:.6f} {w_norm:.6f} {h_norm:.6f}\n")
            logging.info(f"Annotations saved for {frame_file} at {annotation_path}")
            self.update_status.emit(f"Аннотации сохранены: {frame_file}")
        except Exception as e:
            logging.error(f"Failed to save annotations: {str(e)}")
            self.update_status.emit(f"Ошибка сохранения: {str(e)}")

    def confirm_detection(self):
        if not self.current_frame_path or not self.current_annotations:
            logging.warning("No frame or annotations to confirm")
            return
        self.save_annotations()
        self.show_next_frame()

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
            os.rename(self.current_frame_path, negative_path)
            self.current_annotations = []
            logging.info(f"Detection deleted for {frame_file} at {negative_path}")
            self.update_status.emit(f"Детекция удалена: {frame_file}")
            self.show_next_frame()
        except Exception as e:
            logging.error(f"Failed to delete detection: {str(e)}")
            self.update_status.emit(f"Ошибка удаления: {str(e)}")

    def annotate_frame(self):
        self.save_button.setEnabled(self.annotation_mode)
        self.confirm_button.setEnabled(self.review_mode and bool(self.current_annotations))
        self.delete_button.setEnabled(self.review_mode)
        self.next_button.setEnabled(self.annotation_mode or self.review_mode)
        self.prev_button.setEnabled(self.annotation_mode or self.review_mode)
        self.display_frame()

    def show_next_frame(self):
        if self.annotation_mode and self.no_bucket_frames:
            self.current_frame_index = min(self.current_frame_index + 1, len(self.no_bucket_frames) - 1)
            self.current_frame_path = self.no_bucket_frames[self.current_frame_index]
            self.current_annotations = []
            self.image_size = None
            self.display_frame()
        elif self.review_mode and self.low_conf_frames:
            self.current_frame_index = min(self.current_frame_index + 1, len(self.low_conf_frames) - 1)
            self.current_frame_path, self.current_annotations = self.low_conf_frames[self.current_frame_index]
            self.image_size = None
            self.display_frame()

    def show_prev_frame(self):
        if self.annotation_mode and self.no_bucket_frames:
            self.current_frame_index = max(self.current_frame_index - 1, 0)
            self.current_frame_path = self.no_bucket_frames[self.current_frame_index]
            self.current_annotations = []
            self.image_size = None
            self.display_frame()
        elif self.review_mode and self.low_conf_frames:
            self.current_frame_index = max(self.current_frame_index - 1, 0)
            self.current_frame_path, self.current_annotations = self.low_conf_frames[self.current_frame_index]
            self.image_size = None
            self.display_frame()

    def show_frame(self, index: int, review: bool):
        self.current_frame_index = index
        self.review_mode = review
        if review and self.low_conf_frames:
            self.current_frame_path, self.current_annotations = self.low_conf_frames[index]
        elif self.no_bucket_frames:
            self.current_frame_path = self.no_bucket_frames[index]
            self.current_annotations = []
        self.image_size = None
        self.display_frame()