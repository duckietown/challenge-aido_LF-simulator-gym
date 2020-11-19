__version__ = '6.0.42'

from zuper_commons.logs import ZLogger

logger = ZLogger('gym_bridge')


logger.debug(f'gym_bridge version {__version__} path {__file__}')
