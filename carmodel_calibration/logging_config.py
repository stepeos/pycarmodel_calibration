"""configures the logging"""
import logging
import sys
import warnings

LOGGING_CONFIG = 0
_LOGGER = None
def configure_logging(level="info"):
    """
    configure logging
    :param level:       [`debug`, `info`]
    """
    if level.lower() == "debug":
        level = logging.DEBUG
    elif level.lower() == "info":
        level = logging.INFO
    else:
        raise ValueError("Value must be in [`debug`, `info`]")
    # pylint: disable=W0603
    global LOGGING_CONFIG
    global _LOGGER
    warnings.simplefilter("once", UserWarning)
    if LOGGING_CONFIG == 0:
        logging.basicConfig(
            stream=sys.stdout,
            format='%(asctime)s %(levelname)-8s %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            level=level)
        LOGGING_CONFIG = 1
        return
    if not _LOGGER:
        _LOGGER = logging.getLogger(__name__)
    _LOGGER.setLevel(level)
