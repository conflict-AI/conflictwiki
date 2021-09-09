import logging
import typing as Any


def setup_custom_logger(logger_name: str, logging_level: str) -> Any:

    logging_levels = {"NOTSET": logging.NOTSET, "DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING, \
                      "ERROR": logging.ERROR, "CRITICAL": logging.CRITICAL}

    ## formatter
    formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

    ## handlers
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    ## logger
    logger = logging.getLogger()
    logger.setLevel(logging_levels[logging_level])

    logger.addHandler(stream_handler)
    logger.info(f"logger name: {logger_name}, logging_level: {logging_level}")

    return logger