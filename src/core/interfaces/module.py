from typing import Optional

class ModuleInterface:
    """Абстрактный интерфейс для модулей приложения."""
    
    def start(self) -> None:
        """Запуск модуля."""
        pass