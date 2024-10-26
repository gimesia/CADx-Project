import numpy as np
import cv2
import time
import logging

def norm(image: np.ndarray, dtype):
    match dtype:
        case np.uint8:
            return cv2.normalize(image, image, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        case np.uint16:
            return cv2.normalize(image, image, 0, 65535, cv2.NORM_MINMAX).astype(np.uint16)
        case np.float32:
            return cv2.normalize(image, image, 0, 1, cv2.NORM_MINMAX).astype(np.float32)
        case _:
            raise TypeError(f'Unexpected type {dtype}')


# Set up logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log_runtime(func):
    """
    Decorator that logs the runtime of the function it decorates.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)  # Call the original function
        end_time = time.time()
        runtime = end_time - start_time
        logger.info("Function '%s' executed in %.4f seconds", func.__name__, runtime)
        return result
    return wrapper
