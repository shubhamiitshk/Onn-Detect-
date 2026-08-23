import logging
from logging.handlers import RotatingFileHandler
import os
import sys

def get_logger(name: str = "onndetect") -> logging.Logger:
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler (Rotating)
        log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../logs"))
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "training.log")
        
        file_handler = RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.propagate = False
        
    return logger
