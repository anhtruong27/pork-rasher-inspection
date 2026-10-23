"""Occlusion sensitivity maps for investigating model errors.

Adapted from my ECE4179 Assignment 1 (Q4-7). A grey patch is slid over the image and
the model's score for one defect class is recomputed each time. Regions where covering
the image causes a large **drop** in score are the regions the model relies on.

Used here to check whether the models look at the actual defect (e.g. the seal edge
for ``unsealed``) or at irrelevant cues such as glare on the plastic film.

**Colour convention:** `occlusion_map` returns the *drop* in score
(original - occluded), so HIGH values = important regions.
"""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np


def occlusion_map(score_fn: Callable[[np.ndarray], float], image: np.ndarray,
                  patch: int = 60, stride: int = 30, fill: int = 127
                  ) -> Tuple[np.ndarray, float]:
    """Compute an occlusion-sensitivity map.

    Parameters
    ----------
    score_fn : callable
        Maps an RGB uint8 image (H, W, 3) to a scalar score in [0, 1] for the class
        of interest (works for both the CNN and the detector).
    image : numpy.ndarray
        RGB uint8 image (H, W, 3).
    patch : int
        Side of the square occluding patch in pixels.
    stride : int
        Step between patch positions in pixels.
    fill : int
        Grey level of the patch (0-255).

    Returns
    -------
    heatmap : numpy.ndarray
        Shape (H, W): mean score drop over all patches covering each pixel.
        HIGH = important.
    base_score : float
        Score of the un-occluded image.

    Raises
    ------
    ValueError
        If ``patch`` or ``stride`` is not positive or the patch is larger than the image.
    """
    if patch <= 0 or stride <= 0:
        raise ValueError("patch and stride must be positive")
    h, w = image.shape[:2]
    if patch > min(h, w):
        raise ValueError("patch larger than image")
    base = float(score_fn(image))
    acc = np.zeros((h, w), dtype=float)
    cnt = np.zeros((h, w), dtype=float)
    ys = list(range(0, h - patch + 1, stride))
    xs = list(range(0, w - patch + 1, stride))
    if ys[-1] != h - patch:
        ys.append(h - patch)
    if xs[-1] != w - patch:
        xs.append(w - patch)
    for y in ys:
        for x in xs:
            occ = image.copy()
            occ[y:y + patch, x:x + patch] = fill
            drop = base - float(score_fn(occ))
            acc[y:y + patch, x:x + patch] += drop
            cnt[y:y + patch, x:x + patch] += 1
    return acc / np.maximum(cnt, 1), base


def overlay(image: np.ndarray, heatmap: np.ndarray, ax=None, alpha: float = 0.5,
            title: str = "", cmap: str = "jet"):
    """Plot a heatmap over an image with a labelled colour bar.

    Parameters
    ----------
    image : numpy.ndarray
        RGB uint8 image.
    heatmap : numpy.ndarray
        Same height and width as ``image`` (output of `occlusion_map`).
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created if None.
    alpha : float
        Heatmap opacity.
    title : str
        Axes title.
    cmap : str
        Matplotlib colour map.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4.5))
    ax.imshow(image)
    im = ax.imshow(heatmap, cmap=cmap, alpha=alpha)
    plt.colorbar(im, ax=ax, fraction=0.035, label="score drop (high = important)")
    ax.set_title(title)
    ax.axis("off")
    return ax
