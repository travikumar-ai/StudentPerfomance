import os
import sys
import logging
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime("%d_%m_%Y")}.log"

logs_path = os.path.join(os.getcwd(), 'logs')
os.makedirs(logs_path, exist_ok=True)

LOG_FILE_path = os.path.join(
    logs_path, LOG_FILE
)

format = "[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s"

logging.basicConfig(
    filename= LOG_FILE_path,
    format= format,
    level=logging.INFO
)

if __name__ == "__main__":
    logging.info('This is test log message')