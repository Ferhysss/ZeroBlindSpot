import yaml
from typing import Dict, Any
from pathlib import Path

class Config:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = self._load_config(config_path)

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        default_config = {
            "device": "auto",
            "frame_rate": 1,
            "cnn_input_size": [224, 224],
            "use_cnn": False,
            "yolo_conf_threshold": 0.5,
            "project_name": "default",
            "last_project": "",
            "excavator": "Экскаватор A",
            "bucket_volume": 1.5
        }
        try:
            with open(config_path, "r", encoding='utf-8') as f:
                config = yaml.safe_load(f) or default_config
        except FileNotFoundError:
            config = default_config
            Path(config_path).parent.mkdir(exist_ok=True)
            with open(config_path, "w", encoding='utf-8') as f:
                yaml.safe_dump(config, f, allow_unicode=True)
        return config

    def get(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)

    def update(self, key: str, value: Any) -> None:
        self.config[key] = value
        with open("config/config.yaml", "w", encoding='utf-8') as f:
            yaml.safe_dump(self.config, f, allow_unicode=True)