import logging as _logging
from .logging import configure_logger as configure_logger
from .logging import logger as _logger

if not any(not isinstance(h, _logging.NullHandler) for h in _logger.handlers):
    configure_logger()

from .romav2 import RoMaV2 as RoMaV2
