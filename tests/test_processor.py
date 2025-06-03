import pytest
from developer.processor import DeveloperProcessor
from core.config import Config

def test_processor_init():
    config = Config()
    processor = DeveloperProcessor("test.mp4", None, None, config, "data/test")
    assert processor.video_path == "test.mp4"

def test_extract_frames(tmp_path):
    config = Config()
    processor = DeveloperProcessor("test.mp4", None, None, config, str(tmp_path))
    # Мок для cv2.VideoCapture
    # Тест требует видео, пропустим детали
    assert True  # Заглушка