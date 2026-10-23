"""Turn model outputs into a tray-level inspection decision.

Both models are reduced to the same output so they can be compared fairly:
a **score matrix** of shape (n_images, n_classes) where entry (i, c) is the model's
confidence that defect class c is present anywhere in tray i.

- Detector: score = highest confidence of any box of class c (0 if none).
- Baseline CNN: score = sigmoid output for class c.

A tray FAILS if any class score is at or above the threshold for that class.
"""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple, Union

import numpy as np

ArrayLike = Union[float, Sequence[float], np.ndarray]


def scores_from_detections(
    detections: Iterable[Tuple[int, float]], n_classes: int
) -> np.ndarray:
    """Reduce one image's detections to per-class presence scores.

    Parameters
    ----------
    detections : iterable of (int, float)
        ``(class_id, confidence)`` for every predicted box in the image.
    n_classes : int
        Number of defect classes.

    Returns
    -------
    numpy.ndarray
        Shape (n_classes,): maximum confidence per class, 0 where no box was found.

    Raises
    ------
    ValueError
        If a class id is out of range or a confidence is outside [0, 1].
    """
    scores = np.zeros(n_classes, dtype=float)
    for cls, conf in detections:
        cls = int(cls)
        if not 0 <= cls < n_classes:
            raise ValueError(f"class id {cls} out of range for {n_classes} classes")
        if not 0.0 <= conf <= 1.0:
            raise ValueError(f"confidence {conf} outside [0, 1]")
        scores[cls] = max(scores[cls], conf)
    return scores


def presence_from_scores(scores: np.ndarray, threshold: ArrayLike = 0.5) -> np.ndarray:
    """Threshold presence scores.

    Parameters
    ----------
    scores : numpy.ndarray
        Shape (n_classes,) or (n_images, n_classes), values in [0, 1].
    threshold : float or array-like
        One threshold for all classes, or one per class.

    Returns
    -------
    numpy.ndarray
        Boolean array, same shape as ``scores``.
    """
    scores = np.asarray(scores, dtype=float)
    thr = np.asarray(threshold, dtype=float)
    return scores >= thr


def tray_fails(scores: np.ndarray, threshold: ArrayLike = 0.5) -> np.ndarray:
    """Tray-level FAIL decision.

    Parameters
    ----------
    scores : numpy.ndarray
        Shape (n_classes,) for one tray or (n_images, n_classes).
    threshold : float or array-like
        Decision threshold(s).

    Returns
    -------
    bool or numpy.ndarray
        True where the tray FAILS inspection (at least one defect present).
    """
    return presence_from_scores(scores, threshold).any(axis=-1)


def decision_label(scores: np.ndarray, class_names: Sequence[str],
                   threshold: ArrayLike = 0.5) -> str:
    """Human-readable decision for a single tray.

    Parameters
    ----------
    scores : numpy.ndarray
        Shape (n_classes,).
    class_names : sequence of str
        Names in class-id order.
    threshold : float or array-like
        Decision threshold(s).

    Returns
    -------
    str
        e.g. ``"FAIL: unsealed (0.91), wrinkle (0.62)"`` or ``"PASS"``.
    """
    present = presence_from_scores(scores, threshold)
    if not present.any():
        return "PASS"
    found = [f"{class_names[i]} ({scores[i]:.2f})" for i in np.flatnonzero(present)]
    return "FAIL: " + ", ".join(found)
