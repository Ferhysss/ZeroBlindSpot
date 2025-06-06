from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QPushButton, QDialog, QLineEdit, QLabel, QFileDialog, QApplication
from PyQt5.QtCore import Qt
from developer.ui.main_window import DeveloperModule
import os
import yaml
import sys
import logging

class CreateProjectDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Создать проект")
        self.setGeometry(200, 200, 400, 300)
        layout = QVBoxLayout()

        self.status_label = QLabel("Введите данные проекта")
        layout.addWidget(self.status_label)

        self.project_name_input = QLineEdit()
        self.project_name_input.setPlaceholderText("Имя проекта")
        layout.addWidget(self.project_name_input)

        self.video_button = QPushButton("Выбрать видео")
        self.video_button.clicked.connect(self.select_video)
        layout.addWidget(self.video_button)

        self.frame_rate_input = QLineEdit("1")
        self.frame_rate_input.setPlaceholderText("Частота кадров (fps)")
        layout.addWidget(self.frame_rate_input)

        self.confirm_button = QPushButton("Создать")
        self.confirm_button.clicked.connect(self.accept)
        self.confirm_button.setEnabled(False)
        layout.addWidget(self.confirm_button)

        self.video_path = ""
        self.setLayout(layout)

    def select_video(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Выберите видео", "", "Video Files (*.mp4 *.avi)")
        if file_name:
            self.video_path = file_name
            self.status_label.setText(f"Видео: {os.path.basename(file_name)}")
            self.confirm_button.setEnabled(bool(self.project_name_input.text()))

class OpenProjectDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Открыть проект")
        self.setGeometry(200, 200, 400, 200)
        layout = QVBoxLayout()

        self.status_label = QLabel("Выберите папку проекта")
        layout.addWidget(self.status_label)

        self.open_button = QPushButton("Выбрать папку")
        self.open_button.clicked.connect(self.select_project)
        layout.addWidget(self.open_button)

        self.confirm_button = QPushButton("Открыть")
        self.confirm_button.clicked.connect(self.accept)
        self.confirm_button.setEnabled(False)
        layout.addWidget(self.confirm_button)

        self.project_dir = ""
        self.setLayout(layout)

    def select_project(self):
        project_dir = QFileDialog.getExistingDirectory(self, "Выберите папку проекта", "data")
        if project_dir and os.path.exists(os.path.join(project_dir, "project.yaml")):
            self.project_dir = project_dir
            self.status_label.setText(f"Проект: {os.path.basename(project_dir)}")
            self.confirm_button.setEnabled(True)
        else:
            self.status_label.setText("Ошибка: конфигурация проекта не найдена")

class ProjectManager(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ZeroBlindSpot - Управление проектами")
        self.setGeometry(200, 200, 300, 150)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        self.new_project_button = QPushButton("Создать новый проект")
        self.new_project_button.clicked.connect(self.create_project)
        layout.addWidget(self.new_project_button)

        self.open_project_button = QPushButton("Открыть существующий проект")
        self.open_project_button.clicked.connect(self.open_project)
        layout.addWidget(self.open_project_button)

        try:
            with open("config/styles.qss", "r", encoding='utf-8') as f:
                self.setStyleSheet(f.read())
        except FileNotFoundError:
            pass

    def create_project(self):
        dialog = CreateProjectDialog()
        if dialog.exec_():
            project_name = dialog.project_name_input.text().strip()
            if not project_name or not dialog.video_path:
                return
            project_dir = f"data/{project_name}"
            os.makedirs(project_dir, exist_ok=True)
            project_config = {
                "video_path": dialog.video_path,
                "excavator": "Экскаватор A",  # Значение по умолчанию
                "frame_rate": float(dialog.frame_rate_input.text()),
                "project_dir": project_dir
            }
            with open(os.path.join(project_dir, "project.yaml"), "w", encoding='utf-8') as f:
                yaml.safe_dump(project_config, f)
            self.developer_module = DeveloperModule(project_dir=project_dir, video_path=dialog.video_path)
            self.developer_module.start()
            self.hide()

    def open_project(self):
        dialog = OpenProjectDialog()
        if dialog.exec_() and dialog.project_dir:
            project_config_path = os.path.join(dialog.project_dir, "project.yaml")
            with open(project_config_path, "r", encoding='utf-8') as f:
                project_config = yaml.safe_load(f)
            video_path = project_config.get("video_path")
            if video_path and os.path.exists(video_path):
                self.developer_module = DeveloperModule(project_dir=dialog.project_dir, video_path=video_path)
                self.developer_module.start()
                self.hide()
            else:
                dialog.status_label.setText("Ошибка: видео не найдено")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    manager = ProjectManager()
    manager.show()
    sys.exit(app.exec_())