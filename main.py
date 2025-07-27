from RAW.utils import Logger
from pathlib import Path
import time

def main():
    logger = Logger(log_file=Path("app.log"))
    logger.info(content="app started")
    logger.stop()

if __name__ == '__main__':
    main()