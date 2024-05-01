import os

import logger


local_logger = logger.get_logger(__name__)


def check_and_create_dir(dirpath: str) -> None:
    """
    Check if the directory exists, if not, create it.

    Args:
        dirpath (str): Directory path

    Returns:
        None
    """

    if not os.path.isdir(dirpath):
        os.mkdir(dirpath)
    local_logger.info("Created directory: %s", dirpath)
