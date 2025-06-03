from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton
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
        self.class_id: int = 0  # По умолчанию класс "bucket"

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)

        self.save_button = QPushButton("Сохранить аннотацию")
        self.save_button.clicked.connect(self.save_annotations)
        self.save_button.setEnabled(False)
        layout.addWidget(self.save_button)

        self.next_button = QPushButton("Следующий кадр")
        self.next_button.clicked.connect(self.show_next_frame)
        layout.addWidget(self.next_button)

        self.prev_button = QPushButton("Предыдущий кадр")
        self.prev_button.clicked.connect(self.show_prev_frame)
        layout.addWidget(self.prev_button)

        self.setLayout(layout)
        self.image_label.mousePressEvent = self.mouse_press
        self.image_label.mouseMoveEvent = self.mouse_move
        self.image_label.mouseReleaseEvent = self.mouse_release

    def update_frame(self, frame_path: str, annotations: List[Tuple[float, float, float, float, float]]):
        self.current_frame_path = frame_path
        self.current_annotations = annotations
        if frame_path:
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
        scale = min(self.width() / w, self.height() / h)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

        for x, y, w, h, conf in self.current_annotations:
            x1 = int((x - w / 2) * scale)
            y1 = int((y - h / 2) * scale)
            x2 = int((x + w / 2) * scale)
            y2 = int((y + h / 2) * scale)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"conf: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if self.drawing and self.start_point and self.end_point:
            x1, y1 = self.start_point.x(), self.start_point.y()
            x2, y2 = self.end_point.x(), self.end_point.y()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        q_image = QImage(frame.data, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(q_image))

    def mouse_press(self, event):
        if self.annotation_mode and event.button() == Qt.LeftButton:
            self.start_point = event.pos()
            self.drawing = True

    def mouse_move(self, event):
        if self.annotation_mode and self.drawing:
            self.end_point = event.pos()
            self.display_frame()

    def mouse_release(self, event):
        if self.annotation_mode and event.button() == Qt.LeftButton:
            self.end_point = event.pos()
            self.drawing = False
            self.add_annotation()
            self.display_frame()

    def add_annotation(self):
        if self.start_point and self.end_point:
            frame = cv2.imread(self.current_frame_path)
            img_h, img_w = frame.shape[:2]
            scale = min(self.width() / img_w, self.height() / img_h)
            x1, y1 = self.start_point.x() / scale, self.start_point.y() / scale
            x2, y2 = self.end_point.x() / scale, self.end_point.y() / scale
            x = (x1 + x2) / 2
            y = (y1 + y2) / 2
            w = abs(x2 - x1)
            h = abs(y2 - y1)
            self.current_annotations.append((x, y, w, h, 1.0))  # conf=1.0 для ручной аннотации
            self.start_point = None
            self.end_point = None
            self.save_button.setEnabled(True)

    def save_annotations(self):
        if not self.current_frame_path or not self.current_annotations:
            return
        frame_file = os.path.basename(self.current_frame_path)
        output_dir = os.path.dirname(os.path.dirname(self.current_frame_path))
        os.makedirs(f"{output_dir}/annotations", exist_ok=True)
        with open(f"{output_dir}/annotations/{frame_file}.txt", "w", encoding='utf-8') as f:
            for x, y, w, h, conf in self.current_annotations:
                img = cv2.imread(self.current_frame_path)
                img_h, img_w = img.shape[:2]
                x_norm = x / img_w
                y_norm = y / img_h
                w_norm = w / img_w
                h_norm = h / img_h
                f.write(f"{self.class_id} {x_norm:.6f} {y_norm:.6f} {w_norm:.6f} {h_norm:.6f}\n")
        logging.info(f"Annotations saved for {frame_file}")
        self.update_status.emit(f"Аннотации сохранены: {frame_file}")

    def annotate_frame(self):
        self.save_button.setEnabled(True)
        self.display_frame()

    def show_next_frame(self):
        if self.annotation_mode and self.no_bucket_frames:
            self.current_frame_index = min(self.current_frame_index + 1, len(self.no_bucket_frames) - 1)
            self.current_frame_path = self.no_bucket_frames[self.current_frame_index]
            self.current_annotations = []
            self.display_frame()

    def show_prev_frame(self):
        if self.annotation_mode and self.no_bucket_frames:
            self.current_frame_index = max(self.current_frame_index - 1, 0)
            self.current_frame_path = self.no_bucket_frames[self.current_frame_index]
            self.current_annotations = []
            self.display_frame()

    def show_frame(self, index: int, review: bool):
        self.current_frame_index = index
        self.review_mode = review
        if review and self.low_conf_frames:
            self.current_frame_path, self.current_annotations = self.low_conf_frames[index]
        elif self.no_bucket_frames:
            self.current_frame_path = self.no_bucket_frames[index]
            self.current_annotations = []
        self.display_frame()