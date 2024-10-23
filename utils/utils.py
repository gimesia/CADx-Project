import numpy as np
import cv2

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
