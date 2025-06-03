import os
import logging
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime

def setup_logging(project_name: str = "default") -> None:
    try:
        log_dir = f"logs/{project_name}"
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        log_file = f"{log_dir}/app_{timestamp}.log"
        handler = TimedRotatingFileHandler(log_file, when="midnight", backupCount=30, encoding='utf-8')
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
        logging.info("Logging initialized")
    except Exception as e:
        print(f"Logging setup failed: {str(e)}")