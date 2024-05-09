import logging
import time


def monitor_execution_time():
    logger = logging.getLogger('performance_monitor')
    def decorator(func):
        def wrapper(*args, **kwargs):
            st_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - st_time
            logger.info(execution_time)
            return result
        return wrapper
    return decorator
