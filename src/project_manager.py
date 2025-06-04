from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLineEdit, QComboBox, QLabel, QApplication
from PyQt5.QtCore import Qt
from developer.ui.main_window import DeveloperModule
import os
import yaml
import sys
import logging

class ProjectManager(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ZeroBlindSpot - Управление проектами")
        self.setGeometry(200, 200, 400, 300)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        self.status_label = QLabel("Выберите действие")
        layout.addWidget(self.status_label)

        self.new_project_button = QPushButton("Создать новый проект")
        self.new_project_button.clicked.connect(self.create_project)
        layout.addWidget(self.new_project_button)

        self.open_project_button = QPushButton("Открыть существующий проект")
        self.open_project_button.clicked.connect(self.open_project)
        layout.addWidget(self.open_project_button)

        self.project_name_input = QLineEdit()
        self.project_name_input.setPlaceholderText("Имя проекта")
        layout.addWidget(self.project_name_input)

        self.video_button = QPushButton("Выбрать видео")
        self.video_button.clicked.connect(self.select_video)
        layout.addWidget(self.video_button)

        self.excavator_combo = QComboBox()
        self.excavator_combo.addItems(["Экскаватор A", "Экскаватор B", "Экскаватор C"])
        layout.addWidget(self.excavator_combo)

        self.frame_rate_input = QLineEdit("1")
        self.frame_rate_input.setPlaceholderText("Частота кадров (fps)")
        layout.addWidget(self.frame_rate_input)

        self.confirm_button = QPushButton("Подтвердить")
        self.confirm_button.clicked.connect(self.confirm_project)
        self.confirm_button.setEnabled(False)
        layout.addWidget(self.confirm_button)

        self.video_path = ""
        try:
            with open("config/styles.qss", "r", encoding='utf-8') as f:
                self.setStyleSheet(f.read())
        except FileNotFoundError:
            pass

    def select_video(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Выберите видео", "", "Video Files (*.mp4 *.avi)")
        if file_name:
            self.video_path = file_name
            self.status_label.setText(f"Видео: {os.path.basename(file_name)}")
            self.confirm_button.setEnabled(bool(self.project_name_input.text()))

    def create_project(self):
        self.project_name_input.setEnabled(True)
        self.video_button.setEnabled(True)
        self.excavator_combo.setEnabled(True)
        self.frame_rate_input.setEnabled(True)
        self.status_label.setText("Введите имя проекта и выберите видео")

    def open_project(self):
        project_dir = QFileDialog.getExistingDirectory(self, "Выберите папку проекта", "data")
        if project_dir:
            project_config_path = os.path.join(project_dir, "project.yaml")
            if os.path.exists(project_config_path):
                with open(project_config_path, "r", encoding='utf-8') as f:
                    project_config = yaml.safe_load(f)
                video_path = project_config.get("video_path")
                if video_path and os.path.exists(video_path):
                    self.developer_module = DeveloperModule(project_dir=project_dir, video_path=video_path)
                    self.developer_module.start()
                    self.hide()
                else:
                    self.status_label.setText("Ошибка: видео не найдено")
            else:
                self.status_label.setText("Ошибка: конфигурация проекта не найдена")

    def confirm_project(self):
        project_name = self.project_name_input.text().strip()
        if not project_name or not self.video_path:
            self.status_label.setText("Укажите имя проекта и видео")
            return
        project_dir = f"data/{project_name}"
        os.makedirs(project_dir, exist_ok=True)
        project_config = {
            "video_path": self.video_path,
            "excavator": self.excavator_combo.currentText(),
            "frame_rate": float(self.frame_rate_input.text()),
            "project_dir": project_dir
        }
        with open(os.path.join(project_dir, "project.yaml"), "w", encoding='utf-8') as f:
            yaml.safe_dump(project_config, f)
        self.developer_module = DeveloperModule(project_dir=project_dir, video_path=self.video_path)
        self.developer_module.start()
        self.hide()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    manager = ProjectManager()
    manager.show()
    sys.exit(app.exec_())