"""
Logging configuration for the CVM pipeline.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from datetime import datetime

from config import settings

class PipelineLogger:
    """Configure and manage logging for the pipeline."""
    
    @staticmethod
    def setup(name: str = "cvm_pipeline", log_level: str = "INFO") -> logging.Logger:
        """
        Set up logging with file and console handlers.
        
        Args:
            name: Logger name
            log_level: Logging level
            
        Returns:
            Configured logger
        """
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_formatter = logging.Formatter(
            '%(levelname)s - %(message)s'
        )
        
        # File handler (rotating)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = settings.LOGS_DIR / f"cvm_pipeline_{timestamp}.log"
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=settings.LOG_ROTATION_SIZE,
            backupCount=settings.LOG_BACKUP_COUNT,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(console_formatter)
        
        # Add handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        # Log startup
        logger.info("=" * 60)
        logger.info("CVM PIPELINE STARTED")
        logger.info(f"Log file: {log_file}")
        logger.info(f"Log level: {log_level}")
        logger.info("=" * 60)
        
        return logger
    
    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        """
        Get a logger instance.
        
        Args:
            name: Logger name
            
        Returns:
            Logger instance
        """
        return logging.getLogger(name)