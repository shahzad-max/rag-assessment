"""
Logging configuration for the RAG system
"""

import sys
import os
from loguru import logger
from pathlib import Path

try:
    from config import settings
    log_level = settings.log_level if settings else "INFO"
except:
    log_level = os.getenv('LOG_LEVEL', 'INFO')


def setup_logger():
    """Configure logger with appropriate settings"""
    
    # Remove default handler
    logger.remove()
    
    # Add console handler with formatting
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=log_level,
        colorize=True
    )
    
    # Add file handler
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logger.add(
        log_dir / "rag_system_{time:YYYY-MM-DD}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=log_level,
        rotation="00:00",  # Rotate at midnight
        retention="30 days",  # Keep logs for 30 days
        compression="zip"  # Compress old logs
    )
    
    return logger


# Initialize logger
log = setup_logger()

