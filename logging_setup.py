import logging
import sys
from pathlib import Path


def setup_logging(log_dir_base: str):
    log_dir = Path.cwd().joinpath(log_dir_base)
    log_path = log_dir.joinpath("myapp.log")
    log_dir.mkdir(parents=True, exist_ok=True)

    # 1. Reset root logger and set it to WARNING to silence JAX/libraries
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers[:]:
            root.removeHandler(handler)
    root.setLevel(logging.WARNING)

    # 2. Create a dedicated logger for your application code
    logger = logging.getLogger("qaoa")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # Prevent messages from reaching the noisy root logger

    # 3. Setup File Handler (DEBUG and higher to file)
    file_handler = logging.FileHandler(log_path, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M')
    file_handler.setFormatter(file_formatter)

    # 4. Setup Console Handler (INFO and higher to stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console_handler.setFormatter(console_formatter)

    # 5. Add handlers to your specific logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info("Logging initialized. Application logs are isolated from JAX.")
    logger.debug("This debug message will only appear in the file.")