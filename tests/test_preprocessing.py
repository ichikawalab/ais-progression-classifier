import numpy as np

from ais_progression.preprocessing import apply_clahe, pad_to_square


def test_pad_to_square_landscape():
    img = np.full((10, 20), 128, dtype=np.uint8)
    out = pad_to_square(img)
    assert out.shape == (20, 20)
    assert out.dtype == np.uint8
    # Input content should be centered vertically.
    assert np.all(out[5:15, :] == 128)
    assert np.all(out[0:5, :] == 0)
    assert np.all(out[15:20, :] == 0)


def test_pad_to_square_portrait():
    img = np.full((20, 10), 200, dtype=np.uint8)
    out = pad_to_square(img)
    assert out.shape == (20, 20)
    assert np.all(out[:, 5:15] == 200)


def test_pad_to_square_odd_size():
    img = np.full((7, 10), 50, dtype=np.uint8)
    out = pad_to_square(img)
    assert out.shape == (10, 10)


def test_apply_clahe_preserves_shape_and_dtype():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, size=(64, 64), dtype=np.uint8)
    out = apply_clahe(img)
    assert out.shape == img.shape
    assert out.dtype == img.dtype
