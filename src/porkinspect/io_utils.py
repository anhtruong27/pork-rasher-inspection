"""Input validation for tray images.

Every image entering the pipeline goes through `load_image`, so invalid, missing or
unsupported inputs are handled in exactly one place.

Accepted input
--------------
- A path to a ``.jpg``/``.jpeg``/``.png``/``.bmp`` file, or a NumPy array.
- Colour images (H x W x 3, RGB). Greyscale (H x W or H x W x 1) is converted to
  3 channels; RGBA (H x W x 4) has its alpha channel dropped.
- Pixel values ``uint8`` in [0, 255], or float in [0, 1] (converted to uint8).
- Any size; images not equal to the expected 720 x 540 (W x H) are resized and a flag
  is returned so the caller can log it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union

import numpy as np
from PIL import Image, UnidentifiedImageError

#: Expected image size (width, height) in pixels, as exported by Roboflow.
EXPECTED_SIZE: Tuple[int, int] = (720, 540)

#: File extensions accepted by `load_image`.
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


class InvalidImageError(ValueError):
    """Raised when an input cannot be interpreted as a tray image."""


def to_rgb_uint8(array: np.ndarray) -> np.ndarray:
    """Convert an image array to an H x W x 3 ``uint8`` RGB array.

    Parameters
    ----------
    array : numpy.ndarray
        Image with shape (H, W), (H, W, 1), (H, W, 3) or (H, W, 4). ``uint8`` values in
        [0, 255] or floating values in [0, 1].

    Returns
    -------
    numpy.ndarray
        Array of shape (H, W, 3) and dtype ``uint8``.

    Raises
    ------
    InvalidImageError
        If the shape, dtype or value range is not supported, or the array contains
        NaN/inf values.
    """
    if not isinstance(array, np.ndarray):
        raise InvalidImageError(f"Expected a numpy array, got {type(array).__name__}")
    if array.ndim == 2:
        array = array[:, :, None]
    if array.ndim != 3:
        raise InvalidImageError(f"Image must be 2-D or 3-D, got shape {array.shape}")
    h, w, c = array.shape
    if h < 32 or w < 32:
        raise InvalidImageError(f"Image too small ({w}x{h}); minimum is 32x32 pixels")
    if c == 1:
        array = np.repeat(array, 3, axis=2)
    elif c == 4:
        array = array[:, :, :3]  # drop alpha
    elif c != 3:
        raise InvalidImageError(f"Unsupported number of channels: {c}")

    if np.issubdtype(array.dtype, np.floating):
        if not np.all(np.isfinite(array)):
            raise InvalidImageError("Image contains NaN or infinite values")
        if array.min() < 0.0 or array.max() > 1.0:
            raise InvalidImageError("Float images must have values in [0, 1]")
        array = np.round(array * 255.0).astype(np.uint8)
    elif array.dtype != np.uint8:
        raise InvalidImageError(f"Unsupported dtype {array.dtype}; use uint8 or float")
    return np.ascontiguousarray(array)


def load_image(
    source: Union[str, Path, np.ndarray],
    expected_size: Tuple[int, int] = EXPECTED_SIZE,
    resize: bool = True,
) -> Tuple[np.ndarray, bool]:
    """Load and validate a tray image.

    Parameters
    ----------
    source : str, pathlib.Path or numpy.ndarray
        Image file path or an image array (see `to_rgb_uint8`).
    expected_size : tuple of int, optional
        Expected (width, height) in pixels. Default (720, 540).
    resize : bool, optional
        If True (default), images of a different size are resized to ``expected_size``
        with bilinear interpolation. If False, they are returned unchanged.

    Returns
    -------
    image : numpy.ndarray
        RGB image, shape (H, W, 3), dtype ``uint8``.
    was_resized : bool
        True if the image had to be resized.

    Raises
    ------
    FileNotFoundError
        If ``source`` is a path that does not exist.
    InvalidImageError
        If the file extension is unsupported, the file is corrupt, or the array is
        invalid.
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise InvalidImageError(f"Unsupported file type '{path.suffix}'")
        try:
            with Image.open(path) as im:
                array = np.asarray(im.convert("RGB"))
        except (UnidentifiedImageError, OSError) as exc:
            raise InvalidImageError(f"Corrupt or unreadable image: {path}") from exc
    else:
        array = source

    image = to_rgb_uint8(array)
    h, w = image.shape[:2]
    was_resized = False
    if resize and (w, h) != tuple(expected_size):
        image = np.asarray(
            Image.fromarray(image).resize(tuple(expected_size), Image.BILINEAR)
        )
        was_resized = True
    return image, was_resized
