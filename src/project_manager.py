from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QMessageBox
from developer.ui.main_window import DeveloperModule
import logging

class ProjectManager(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ZeroBlindSpot - Выбор режима")
        self.developer_module = None
        self._init_ui()

    def _init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        developer_button = QPushButton("Режим разработчика")
        developer_button.clicked.connect(self._start_developer)
        layout.addWidget(developer_button)
        operator_button = QPushButton("Режим оператора")
        operator_button.clicked.connect(self._start_operator)
        layout.addWidget(operator_button)

    def _start_developer(self):
        self.developer_module = DeveloperModule()
        self.developer_module.show()
        self.close()  # Закрываем окно

    def _start_operator(self):
        logging.info("Operator mode not implemented")
        QMessageBox.information(self, "Ошибка", "Режим оператора не реализован")