"""
Shared logging configuration using loguru for consistent formatting across all APIs.
Provides colored, structured logging with consistent format across all Project Friday services.
"""

import sys
from loguru import logger


def setup_loguru_logging(service_name: str = "API", level: str = None):
    """
    Configure loguru with consistent formatting across all services.

    Args:
        service_name: Name of the service (e.g., "Orchestration", "WebsiteAPI", "GraphService")
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL). If None, uses LOG_LEVEL env var.
    """
    import os

    # Use environment variable if level not provided
    if level is None:
        level = os.environ.get("LOG_LEVEL", "INFO")

    # Remove default handler
    logger.remove()

    # Add colored, formatted handler for console output
    logger.add(
        sys.stdout,
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
               "<level>{level: <8}</level> | "
               f"<cyan>{service_name}</cyan> | "
               "<level>{message}</level>",
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    return logger


def configure_uvicorn_logging():
    """
    Configure uvicorn to use loguru instead of its default logging.
    This intercepts uvicorn's logs and routes them through loguru.
    """
    import logging
    
    class InterceptHandler(logging.Handler):
        def emit(self, record):
            # Get corresponding Loguru level if it exists
            try:
                level = logger.level(record.levelname).name
            except ValueError:
                level = record.levelno
            
            # Find caller from where originated the logged message
            frame, depth = logging.currentframe(), 2
            while frame.f_code.co_filename == logging.__file__:
                frame = frame.f_back
                depth += 1
            
            logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())
    
    # Intercept standard logging
    logging.basicConfig(handlers=[InterceptHandler()], level=0)
    
    # Set uvicorn loggers to use our interceptor
    for logger_name in ["uvicorn", "uvicorn.access", "uvicorn.error"]:
        logging_logger = logging.getLogger(logger_name)
        logging_logger.handlers = [InterceptHandler()]
        logging_logger.propagate = False
