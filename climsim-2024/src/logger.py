import logging
import logging.handlers
import os
import sys
from typing import Optional

import env


def get_logger(
    logger_name: str,
    log_level: int = env.LOG_LEVEL,
    log_filepath: Optional[str] = env.LOG_FILEPATH,
) -> logging.Logger:
    """
    Get logger with stream and file handlers (if log_filepath is provided)

    Args:
        logger_name (str): Logger name
        log_level (int): Log level of the logger
        log_filepath (str): Log filepath

    Returns:
        logging.Logger: Logger
    """

    # Initialize logger object
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)

    formatter = logging.Formatter(fmt=env.LOG_FMT, datefmt=env.LOG_DATEFMT)

    # Add stream handler to log to console
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(log_level)
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
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
