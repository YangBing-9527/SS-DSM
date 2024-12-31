CONFIG = {
    'version': 1,
    'disable_existing_loggers': True,
    'formatters': {
        'precise': {
            'format': '%(asctime)s %(levelname)s %(filename)s[line:%(lineno)d] : %(message)s'
        },
        'simple': {
            'format': '%(filename)s %(message)s'
        },
    },
    'handlers': {
        'console': {
            'level':'INFO',
            'class':'logging.StreamHandler',
            'formatter': 'precise'
        },
        'filehandler':{
            'level':'INFO',
            'class':'logging.FileHandler',
            'formatter': 'precise',
            'filename': 'log.log',
            'mode': 'w'
        }
    },
    'loggers': {
        '': {
            'handlers':['console', 'filehandler'],
            'propagate': False,
            'level':'INFO',
        },
        'debuglog': {
            'handlers': ['console', 'filehandler'],
            'propagate': False,
            'level': 'DEBUG',
        }
    }
}