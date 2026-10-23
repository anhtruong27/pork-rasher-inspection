"""Tests for evaluation metrics against hand-calculated values (porkinspect.metrics)."""
import numpy as np
import pytest

from porkinspect.metrics import (best_threshold, binary_counts, multilabel_report, prf,
                                 threshold_sweep)


def test_binary_counts_and_prf_hand_example():
    # tp=2, fp=1, fn=1, tn=1  -> P=2/3, R=2/3, F1=2/3
    c = binary_counts([1, 1, 1, 0, 0], [1, 1, 0, 1, 0])
    assert c == {"tp": 2, "fp": 1, "fn": 1, "tn": 1}
    m = prf(c)
    assert m["precision"] == pytest.approx(2 / 3) and m["f1"] == pytest.approx(2 / 3)


def test_prf_zero_division_is_zero():
    assert prf({"tp": 0, "fp": 0, "fn": 0, "tn": 3}) == {"precision": 0, "recall": 0, "f1": 0}


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        binary_counts([1, 0], [1])


def test_perfect_prediction():
    y = np.array([[1, 0], [0, 1], [0, 0]])
    r = multilabel_report(y, y, ["a", "b"])
    assert r["macro_f1"] == 1 and r["exact_match"] == 1 and r["tray"]["accuracy"] == 1
    assert r["tray"]["balanced_accuracy"] == 1


def test_macro_vs_micro_under_imbalance():
    # class a: 9 positives all found; class b: 1 positive missed
    y = np.zeros((10, 2), int); y[:9, 0] = 1; y[9, 1] = 1
    p = y.copy(); p[9, 1] = 0
    r = multilabel_report(y, p, ["a", "b"])
    assert r["micro"]["recall"] == pytest.approx(0.9)
    assert r["macro_recall"] == pytest.approx(0.5)       # balanced: hides nothing


def test_tray_balanced_accuracy_nan_without_pass_trays():
    y = np.array([[1, 0], [0, 1]])
    r = multilabel_report(y, y, ["a", "b"])
    assert r["tray"]["n_pass_trays"] == 0 and np.isnan(r["tray"]["balanced_accuracy"])


def test_threshold_sweep_monotone_recall():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, (50, 3))
    s = np.clip(y * 0.6 + rng.random((50, 3)) * 0.5, 0, 1)
    sw = threshold_sweep(y, s, np.linspace(0.05, 0.95, 19))
    assert np.all(np.diff(sw["micro_recall"]) <= 1e-12)   # recall never increases
    t = best_threshold(sw)
    assert 0.05 <= t <= 0.95


from porkinspect.metrics import bootstrap_ci


def test_bootstrap_ci_contains_estimate_and_shrinks_with_n():
    rng = np.random.default_rng(1)
    f = lambda t, p: float(np.mean(t == p))
    small_t = rng.integers(0, 2, (40, 1)); small_p = np.where(rng.random((40, 1)) < .8, small_t, 1 - small_t)
    big_t = rng.integers(0, 2, (800, 1)); big_p = np.where(rng.random((800, 1)) < .8, big_t, 1 - big_t)
    a = bootstrap_ci(small_t, small_p, f, n_boot=500)
    b = bootstrap_ci(big_t, big_p, f, n_boot=500)
    assert a["low"] <= a["estimate"] <= a["high"]
    assert (b["high"] - b["low"]) < (a["high"] - a["low"])


def test_bootstrap_ci_rejects_empty():
    with pytest.raises(ValueError):
        bootstrap_ci(np.zeros((0, 2)), np.zeros((0, 2)), lambda t, p: 0.0)


from porkinspect.metrics import (always_fail_predictions, cost_threshold, paired_bootstrap,
                                 tray_cost)


def test_always_fail_rule():
    y = np.array([[1, 0], [0, 0], [0, 1]])
    p = always_fail_predictions(y, 0)
    assert p.tolist() == [[True, False]] * 3
    r = multilabel_report(y, p, ["a", "b"])
    assert r["tray"]["balanced_accuracy"] == pytest.approx(0.5)


def test_tray_cost_hand_example():
    y = np.array([[1], [1], [0], [0]])
    p = np.array([[0], [1], [1], [0]])          # 1 miss, 1 waste
    assert tray_cost(y, p, miss_cost=10, waste_cost=1) == 11
    with pytest.raises(ValueError):
        tray_cost(y, p, -1, 1)


def test_cost_threshold_moves_with_cost_ratio():
    y = np.array([[1], [1], [0], [0]])
    s = np.array([[0.9], [0.3], [0.4], [0.1]])
    th = [0.2, 0.35, 0.5]
    # misses expensive -> low threshold (catch tray with 0.3, waste 0.4 tray)
    assert cost_threshold(y, s, th, miss_cost=10, waste_cost=1)["threshold"] == 0.2
    # waste expensive -> higher threshold
    assert cost_threshold(y, s, th, miss_cost=1, waste_cost=10)["threshold"] == 0.5


def test_paired_bootstrap_detects_clear_difference():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, (100, 1))
    good = y.copy()
    bad = 1 - y
    f = lambda t, p: float(np.mean(t == p))
    r = paired_bootstrap(y, good, bad, f, n_boot=300)
    assert r["diff"] == pytest.approx(1.0) and r["low"] > 0 and r["p_a_not_better"] == 0
    with pytest.raises(ValueError):
        paired_bootstrap(y, good, bad[:5], f)
