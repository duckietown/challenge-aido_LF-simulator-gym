__version__ = "6.1.77"

from zuper_commons.logs import ZLogger

logger = ZLogger(__name__)
import os

path = os.path.dirname(os.path.dirname(__file__))

logger.debug(f"duckietown-symulator-gym version {__version__} path {path}")
from .code import *
