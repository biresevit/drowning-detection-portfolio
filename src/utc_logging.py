import logging
from logging.handlers import RotatingFileHandler
import time
import os


def get_logger(
    log_level: str,
    filename: str,
    log_type: int,
    max_log_size_mb=10,
    num_log_backup_files=5,
):
    if log_type not in (0, 1, 2):
        raise ValueError(
            f"Invalid log type: {log_type}. 0 will log to both log file and console output. 1 will log only to log file. 2 will log to only console"
        )
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError("Invalid log level: %s" % log_level)

    log_dir = os.path.dirname(filename)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    # Create a custom formatter
    class CustomFormatter(logging.Formatter):
        converter = time.gmtime  # Use UTC time

        def formatTime(self, record, datefmt=None):
            if datefmt:
                s = time.strftime(datefmt, self.converter(record.created))
            else:
                s = logging.Formatter.formatTime(self, record, datefmt)
            return s

    log_formatter = CustomFormatter(
        "%(asctime)s - %(levelname)s - %(message)s", "%d-%m-%Y %H:%M:%S"
    )

    # Custom file handler subclass to flush after each emit
    class FlushRotatingFileHandler(RotatingFileHandler):
        def emit(self, record):
            super().emit(record)
            self.flush()  # flush immediately after writing

    file_handler = FlushRotatingFileHandler(
        filename,
        mode="a",
        maxBytes=max_log_size_mb * 1024 * 1024,
        backupCount=num_log_backup_files,
        encoding=None,
        delay=0,
    )
    file_handler.setFormatter(log_formatter)

    # Create a handler for the console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)

    my_logger = logging.getLogger()
    my_logger.setLevel(numeric_level)

    # Remove any existing handlers to avoid duplicate logs if get_logger called multiple times
    if my_logger.hasHandlers():
        my_logger.handlers.clear()

    if log_type == 0:
        my_logger.addHandler(file_handler)
        my_logger.addHandler(console_handler)
    elif log_type == 1:
        my_logger.addHandler(file_handler)
    else:
        my_logger.addHandler(console_handler)

    my_logger.info(f"Log mode {log_type}, log level: {log_level}. Log file: {filename}")
    if log_type in (0, 1):
        file_handler.flush()  # extra flush just after initial log
    return my_logger
