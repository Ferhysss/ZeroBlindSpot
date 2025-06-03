import sys
import argparse
from PyQt5.QtWidgets import QApplication
from core.logging import setup_logging
from core.interfaces.module import ModuleInterface
from developer.ui.main_window import DeveloperModule
from operator_module.ui.main_window import OperatorModule

def main():
    parser = argparse.ArgumentParser(description="ZeroBlindSpot")
    parser.add_argument("--mode", choices=["develop", "operator"], default="develop")
    args = parser.parse_args()

    setup_logging(args.mode)
    app = QApplication(sys.argv)
    module: ModuleInterface = DeveloperModule() if args.mode == "develop" else OperatorModule()
    module.start()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()