from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt5.QtCore import Qt

class ResultViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.result_label = QLabel("Циклы: 0, Выработка: 0.0 м³")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.result_label)

    def update_results(self, cycles: int, output: float) -> None:
        self.result_label.setText(f"Циклы: {cycles}, Выработка: {output:.2f} м³")

    def reset(self) -> None:
        self.result_label.setText("Циклы: 0, Выработка: 0.0 м³")