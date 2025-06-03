from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt5.QtCore import Qt

class ResultViewer(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        self.result_label = QLabel("Результаты обработки видео будут здесь")
        self.result_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.result_label)
        self.setLayout(layout)

    def update_results(self, cycle_count: int):
        self.result_label.setText(f"Обнаружено циклов: {cycle_count}")

    def reset(self):
        self.result_label.setText("Результаты обработки видео будут здесь")