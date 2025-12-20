"""
Logging utilities.
"""

import logging
from datetime import datetime


def setup_logging(log_file: str = "systematix.log") -> logging.Logger:
    """Setup application logging."""
    logger = logging.getLogger("systematix")
    logger.setLevel(logging.DEBUG)

    handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


logger = setup_logging()


def log_pricing_event(model: str, option: str, price: float):
    """Log pricing event."""
    logger.info(f"Priced {option} with {model}: price={price:.6f}")


def log_error(message: str):
    """Log error."""
    logger.error(message)

