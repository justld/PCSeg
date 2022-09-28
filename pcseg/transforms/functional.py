import numpy as np
from typing import Tuple


def normalize(im: np.ndarray,
              mean: Tuple[float, float, float],
              std: Tuple[float, float, float]) -> np.ndarray:
    im -= mean
    im /= std
    return im
