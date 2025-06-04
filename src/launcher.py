from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt
from project_manager import ProjectManager
from operator_module.ui.main_window import OperatorModule
from core.logging import setup_logging
import sys

class Launcher(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ZeroBlindSpot Launcher")
        self.setGeometry(200, 200, 300, 150)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        develop_button = QPushButton("Режим разработчика")
        develop_button.clicked.connect(self.launch_develop)
        layout.addWidget(develop_button)

        operator_button = QPushButton("Режим оператора")
        operator_button.clicked.connect(self.launch_operator)
        layout.addWidget(operator_button)

        try:
            with open("config/styles.qss", "r", encoding='utf-8') as f:
                self.setStyleSheet(f.read())
        except FileNotFoundError:
            pass

    def launch_develop(self):
        setup_logging("develop")
        self.project_manager = ProjectManager()
        self.project_manager.show()
        self.hide()

    def launch_operator(self):
        setup_logging("operator")
        self.operator_module = OperatorModule()
        self.operator_module.start()
        self.hide()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    launcher = Launcher()
    launcher.show()
    sys.exit(app.exec_())