"""Tests for input validation (porkinspect.io_utils)."""
import numpy as np
import pytest
from PIL import Image

from porkinspect.io_utils import InvalidImageError, load_image, to_rgb_uint8


def test_greyscale_converted_to_three_channels():
    out = to_rgb_uint8(np.zeros((64, 64), np.uint8))
    assert out.shape == (64, 64, 3) and out.dtype == np.uint8


def test_rgba_alpha_dropped():
    assert to_rgb_uint8(np.zeros((64, 64, 4), np.uint8)).shape == (64, 64, 3)


def test_float_image_scaled():
    out = to_rgb_uint8(np.ones((40, 40, 3), np.float32))
    assert out.max() == 255


@pytest.mark.parametrize("bad", [
    np.zeros((10, 10, 3), np.uint8),            # too small
    np.zeros((64, 64, 2), np.uint8),            # wrong channels
    np.full((64, 64, 3), 2.0),                  # float out of range
    np.full((64, 64, 3), np.nan),               # NaN
    np.zeros((64, 64, 3), np.int32),            # unsupported dtype
    np.zeros((2, 64, 64, 3), np.uint8),         # 4-D
])
def test_invalid_arrays_rejected(bad):
    with pytest.raises(InvalidImageError):
        to_rgb_uint8(bad)


def test_non_array_rejected():
    with pytest.raises(InvalidImageError):
        to_rgb_uint8([[1, 2], [3, 4]])


def test_wrong_size_is_resized_and_flagged():
    img, resized = load_image(np.zeros((100, 200, 3), np.uint8))
    assert resized and img.shape == (540, 720, 3)


def test_correct_size_not_resized():
    img, resized = load_image(np.zeros((540, 720, 3), np.uint8))
    assert not resized and img.shape == (540, 720, 3)


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_image(tmp_path / "nope.jpg")


def test_unsupported_extension(tmp_path):
    p = tmp_path / "a.gif"
    p.write_bytes(b"GIF89a")
    with pytest.raises(InvalidImageError):
        load_image(p)


def test_corrupt_file(tmp_path):
    p = tmp_path / "broken.jpg"
    p.write_bytes(b"not an image")
    with pytest.raises(InvalidImageError):
        load_image(p)


def test_png_roundtrip(tmp_path):
    p = tmp_path / "ok.png"
    Image.fromarray(np.full((540, 720, 3), 7, np.uint8)).save(p)
    img, resized = load_image(p)
    assert img[0, 0, 0] == 7 and not resized
