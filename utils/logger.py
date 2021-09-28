import sys
from functools import lru_cache

from loguru import logger

logger_format = "<green>{time:MM/DD HH:mm:ss.SS}</green> | <level>{level: ^7}</level> |" \
                "{process.name:<5}.{thread.name:<5}: " \
                "<cyan>{name:<8}</cyan>:<cyan>{function:<5}</cyan>:<cyan>{line:<4}</cyan>" \
                " - <level>{message}</level>"


@lru_cache()
def config_logger():
    logger.remove()

    logger.add(sys.stderr, format=logger_format, backtrace=False, diagnose=False)


config_logger()
