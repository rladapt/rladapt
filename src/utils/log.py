import logging


def get_logger(name, log_path='out.log', level=logging.INFO):
    # create logger with 'spam_application'
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(log_path)
    fh.setLevel(level)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(level)
    # create formatter and add it to the handlers
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger
