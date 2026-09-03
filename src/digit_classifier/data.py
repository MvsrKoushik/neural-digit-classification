import numpy as np


def normalize_images(images):
    values = np.asarray(images)
    if values.ndim not in {3, 4} or values.shape[-2:] != (28, 28):
        if values.ndim != 4 or values.shape[1:] != (28, 28, 1):
            raise ValueError("expected (N, 28, 28) or (N, 28, 28, 1)")
    values = values.astype("float32")
    if values.max(initial=0) > 1:
        values /= 255.0
    if values.ndim == 3:
        values = values[..., None]
    return values

