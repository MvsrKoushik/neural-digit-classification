import numpy as np
import pytest
from digit_classifier import normalize_images


def test_normalizes_and_adds_channel():
    result = normalize_images(np.full((2, 28, 28), 255, dtype=np.uint8))
    assert result.shape == (2, 28, 28, 1) and result.max() == 1


def test_rejects_wrong_shape():
    with pytest.raises(ValueError):
        normalize_images(np.zeros((2, 8, 8)))

