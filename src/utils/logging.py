"""
Structured Logging Configuration
Provides JSON-formatted logging with context for production environments
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from loguru import logger as loguru_logger
from pythonjsonlogger import jsonlogger


def setup_logging(
    log_level: str = "INFO",
    log_dir: Optional[Path] = None,
    console_output: bool = True,
    use_structured_logging: bool = True,
    log_file_name: str = "app.log",
) -> logging.Logger:
    """
    Setup logging configuration for the application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_dir: Directory for log files (None to disable file logging)
        console_output: Whether to output to console
        use_structured_logging: Use JSON formatting for structured logs
        log_file_name: Name of the log file

    Returns:
        Configured logger instance
    """
    # Remove default loguru handler
    loguru_logger.remove()

    # Configure loguru
    if console_output:
        if use_structured_logging:
            loguru_logger.add(
                sys.stdout,
                format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
                level=log_level,
                colorize=True,
            )
        else:
            loguru_logger.add(
                sys.stdout,
                format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
                level=log_level,
                colorize=True,
            )

    # Add file handler if log_dir is specified
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / log_file_name

        if use_structured_logging:
            # JSON format for easy parsing
            loguru_logger.add(
                log_file,
                format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {name}:{function}:{line} | {message}",
                level=log_level,
                rotation="100 MB",
                retention="30 days",
                compression="zip",
                serialize=True,  # JSON format
            )
        else:
            loguru_logger.add(
                log_file,
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
                level=log_level,
                rotation="100 MB",
                retention="30 days",
                compression="zip",
            )

    # Configure standard library logging to use loguru
    class InterceptHandler(logging.Handler):
        def emit(self, record):
            # Get corresponding Loguru level if it exists
            try:
                level = loguru_logger.level(record.levelname).name
            except ValueError:
                level = record.levelno

            # Find caller from where originated the logged message
            frame, depth = logging.currentframe(), 2
            while frame.f_code.co_filename == logging.__file__:
                frame = frame.f_back
                depth += 1

            loguru_logger.opt(depth=depth, exception=record.exc_info).log(
                level, record.getMessage()
            )

    # Intercept standard logging
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

    # Intercept specific loggers
    for logger_name in ["uvicorn", "uvicorn.access", "fastapi"]:
        logging.getLogger(logger_name).handlers = [InterceptHandler()]

    return loguru_logger


def get_logger(name: str):
    """
    Get a logger instance with the given name.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    return loguru_logger.bind(name=name)


class LogContext:
    """Context manager for adding structured context to logs."""

    def __init__(self, **kwargs):
        """
        Initialize log context.

        Args:
            **kwargs: Key-value pairs to add to log context
        """
        self.context = kwargs

    def __enter__(self):
        """Enter context - bind context to logger."""
        loguru_logger.configure(extra=self.context)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context - remove context from logger."""
        loguru_logger.configure(extra={})


# Example usage:
# with LogContext(user_id=123, request_id="abc"):
#     logger.info("Processing request")  # Will include user_id and request_id
