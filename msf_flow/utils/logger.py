import logging

def init_logger(log_level=logging.WARNING):
    """Initialize logger.
    
    Args:
        log_level (int): logging level
            May be (in order of increasing verboseness): 
            logging.CRITICAL or 50
            logging.ERROR or 40
            logging.WARNING or 30
            logging.INFO or 20
            logging.DEBUG or 10
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    logger.addHandler(ch)
    return logger
