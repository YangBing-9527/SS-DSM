import logging.config
from framework.logger.config import CONFIG


def config_logger(filename):
    temp = CONFIG
    temp['handlers']['filehandler']['filename'] = filename
    logging.config.dictConfig(temp)


def getDebugLogger():
    return logging.getLogger('debuglog')
