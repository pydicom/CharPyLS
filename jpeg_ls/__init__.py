import logging

from .CharLS import encode
from .CharLS import decode
#
from .CharLS import write
from .CharLS import read


__version__ = "1.1.0.dev0"


# Setup default logging
_logger = logging.getLogger(__name__)
_logger.addHandler(logging.NullHandler())


def debug_logger() -> None:
    """Setup the logging for debugging."""
    logger = logging.getLogger(__name__)
    logger.handlers = []
    handler = logging.StreamHandler()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(levelname).1s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
