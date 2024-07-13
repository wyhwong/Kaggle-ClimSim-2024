import logging
import logging.handlers
import os
import sys
from typing import Optional

import src.env


def get_logger(
    logger_name: str,
    streaming_log_level: int = src.env.STREAMING_LOG_LEVEL,
    file_log_level: int = src.env.FILE_LOG_LEVEL,
    log_filepath: Optional[str] = src.env.LOG_FILEPATH,
) -> logging.Logger:
    """Get logger with stream and file handlers (if log_filepath is provided)

    Args:
        logger_name (str): Logger name
        streaming_log_level (int): Streaming log level
        file_log_level (int): File log level
        log_filepath (str): Log filepath

    Returns:
        logger (logging.Logger): Logger
    """

    # Initialize logger object
    logger = logging.getLogger(logger_name)
    logger.setLevel(file_log_level)
    formatter = logging.Formatter(fmt=src.env.LOG_FMT, datefmt=src.env.LOG_DATEFMT)

    # Add stream handler to log to console
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(streaming_log_level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_filepath:
        # Check if directory exists
        dirname = os.path.dirname(log_filepath) if os.path.dirname(log_filepath) else "."
        if not os.path.exists(dirname):
            raise FileNotFoundError(f"Directory {dirname} does not exist")

        # Add file handler to log to file
        file_handler = logging.handlers.TimedRotatingFileHandler(
            filename=log_filepath,
            when="midnight",
            backupCount=7,
        )
        file_handler.setLevel(file_log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
