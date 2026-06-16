import logging, sys
from app.core.config import get_settings
settings = get_settings()

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S"))
        logger.addHandler(h)
    logger.setLevel(getattr(logging, settings.log_level))
    return logger
