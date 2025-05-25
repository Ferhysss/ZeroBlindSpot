import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel
from PyQt5.QtCore import Qt

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ZeroBlindSpot")
        self.setGeometry(100, 100, 800, 600)

        # Главный виджет и макет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Кнопка загрузки видео
        self.load_button = QPushButton("Загрузить видео")
        self.load_button.clicked.connect(self.load_video)
        layout.addWidget(self.load_button)

        # Метка для статуса
        self.status_label = QLabel("Ожидание загрузки видео...")
        layout.addWidget(self.status_label)

        # Видеоплеер (заглушка)
        self.video_placeholder = QLabel("Здесь будет видео")
        self.video_placeholder.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.video_placeholder)

        # Применение стилей
        with open("ui/styles.qss", "r") as f:
            self.setStyleSheet(f.read())

    def load_video(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Выберите видео", "", "Video Files (*.mp4 *.avi)")
        if file_name:
            self.status_label.setText(f"Загружено: {file_name}")
            # Здесь будет разбиение видео (добавим позже)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())