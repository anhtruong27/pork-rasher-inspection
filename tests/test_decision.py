"""Tests for tray-level decisions (porkinspect.decision)."""
import numpy as np
import pytest

from porkinspect.decision import (decision_label, presence_from_scores,
                                  scores_from_detections, tray_fails)

NAMES = ["loose-meat", "packaging-error", "twisted-meat", "unsealed", "wrinkle"]


def test_scores_take_max_confidence_per_class():
    s = scores_from_detections([(3, 0.4), (3, 0.9), (0, 0.2)], 5)
    np.testing.assert_allclose(s, [0.2, 0, 0, 0.9, 0])


def test_no_detections_gives_zero_scores():
    assert scores_from_detections([], 5).sum() == 0


@pytest.mark.parametrize("det", [[(5, 0.5)], [(-1, 0.5)], [(0, 1.2)]])
def test_invalid_detections_rejected(det):
    with pytest.raises(ValueError):
        scores_from_detections(det, 5)


def test_threshold_is_inclusive():
    assert presence_from_scores(np.array([0.5, 0.49]), 0.5).tolist() == [True, False]


def test_per_class_thresholds():
    s = np.array([0.6, 0.6])
    assert presence_from_scores(s, [0.5, 0.7]).tolist() == [True, False]


def test_tray_fails_batch():
    s = np.array([[0.1, 0.2, 0, 0, 0], [0, 0, 0, 0.8, 0]])
    assert tray_fails(s, 0.5).tolist() == [False, True]


def test_decision_label():
    assert decision_label(np.zeros(5), NAMES) == "PASS"
    assert decision_label(np.array([0, 0, 0, 0.91, 0.62]), NAMES) == \
        "FAIL: unsealed (0.91), wrinkle (0.62)"
