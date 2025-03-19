from loguru import logger

logger.add("logs/log.log", format="{time} - {function} - {message}")
logger.add("logs/error.log", backtrace=True, diagnose=True, filter=lambda record: record["level"].name == "ERROR")