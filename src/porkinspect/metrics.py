"""Evaluation metrics for tray-level inspection.

Because the defect classes are imbalanced, overall numbers can hide poor performance on
rare classes. We therefore report, for every model:

- per-class precision, recall and F1 (presence of each defect type in a tray);
- macro-averaged recall ("balanced accuracy" over classes) and macro F1;
- micro (pooled) precision/recall/F1;
- exact-match ratio (all five presence flags correct);
- tray PASS/FAIL accuracy and the 2x2 confusion matrix, *only meaningful if the test
  set contains defect-free trays*.

Detector box-level metrics (mAP@0.5, mAP@0.5:0.95) come from Ultralytics' validator.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def binary_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    """Confusion counts for a binary problem.

    Parameters
    ----------
    y_true, y_pred : numpy.ndarray
        Boolean or 0/1 arrays of equal shape.

    Returns
    -------
    dict
        ``{"tp", "fp", "fn", "tn"}``.
    """
    t = np.asarray(y_true).astype(bool)
    p = np.asarray(y_pred).astype(bool)
    if t.shape != p.shape:
        raise ValueError(f"shape mismatch {t.shape} vs {p.shape}")
    return {
        "tp": int(np.sum(t & p)),
        "fp": int(np.sum(~t & p)),
        "fn": int(np.sum(t & ~p)),
        "tn": int(np.sum(~t & ~p)),
    }


def prf(counts: Dict[str, int]) -> Dict[str, float]:
    """Precision, recall and F1 from confusion counts.

    Parameters
    ----------
    counts : dict
        Output of `binary_counts`.

    Returns
    -------
    dict
        ``{"precision", "recall", "f1"}``; 0 when undefined.
    """
    p = _safe_div(counts["tp"], counts["tp"] + counts["fp"])
    r = _safe_div(counts["tp"], counts["tp"] + counts["fn"])
    f = _safe_div(2 * p * r, p + r)
    return {"precision": p, "recall": r, "f1": f}


def multilabel_report(y_true: np.ndarray, y_pred: np.ndarray,
                      class_names: Sequence[str]) -> Dict[str, object]:
    """Full tray-level evaluation.

    Parameters
    ----------
    y_true : numpy.ndarray
        Shape (n_images, n_classes), ground-truth presence (0/1).
    y_pred : numpy.ndarray
        Shape (n_images, n_classes), predicted presence (0/1 or bool).
    class_names : sequence of str
        Names in class order.

    Returns
    -------
    dict
        Keys: ``per_class`` (dict of dicts incl. support), ``macro_recall``
        (balanced accuracy over classes), ``macro_f1``, ``micro`` (dict),
        ``exact_match``, ``tray`` (PASS/FAIL counts, accuracy, balanced accuracy,
        and ``n_pass_trays``).
    """
    t = np.asarray(y_true).astype(bool)
    p = np.asarray(y_pred).astype(bool)
    if t.shape != p.shape or t.ndim != 2 or t.shape[1] != len(class_names):
        raise ValueError("y_true and y_pred must both be (n_images, n_classes)")

    per_class = {}
    for i, name in enumerate(class_names):
        c = binary_counts(t[:, i], p[:, i])
        per_class[name] = {**prf(c), **c, "support": int(t[:, i].sum())}
    # Only classes that occur in the ground truth count towards macro averages.
    present = [n for n in class_names if per_class[n]["support"] > 0]
    macro_recall = float(np.mean([per_class[n]["recall"] for n in present])) if present else 0.0
    macro_f1 = float(np.mean([per_class[n]["f1"] for n in present])) if present else 0.0
    micro = prf(binary_counts(t.ravel(), p.ravel()))
    exact = float(np.mean(np.all(t == p, axis=1)))

    fail_true = t.any(axis=1)
    fail_pred = p.any(axis=1)
    tc = binary_counts(fail_true, fail_pred)  # positive class = FAIL
    tray_acc = _safe_div(tc["tp"] + tc["tn"], len(fail_true))
    rec_fail = _safe_div(tc["tp"], tc["tp"] + tc["fn"])
    rec_pass = _safe_div(tc["tn"], tc["tn"] + tc["fp"])
    n_pass = int((~fail_true).sum())
    tray_bal = (rec_fail + rec_pass) / 2 if n_pass and n_pass < len(fail_true) else float("nan")
    return {
        "per_class": per_class,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "micro": micro,
        "exact_match": exact,
        "tray": {**tc, "accuracy": tray_acc, "fail_recall": rec_fail,
                 "pass_recall": rec_pass, "balanced_accuracy": tray_bal,
                 "n_pass_trays": n_pass},
    }


def threshold_sweep(y_true: np.ndarray, scores: np.ndarray,
                    thresholds: Sequence[float]) -> Dict[str, np.ndarray]:
    """Sensitivity of tray-level metrics to the decision threshold.

    This is the main sensitivity analysis: on a production line a missed defect
    (false PASS) and a wasted good tray (false FAIL) have different costs, and the
    threshold sets the trade-off.

    Parameters
    ----------
    y_true : numpy.ndarray
        Shape (n_images, n_classes), ground-truth presence.
    scores : numpy.ndarray
        Shape (n_images, n_classes), presence scores in [0, 1].
    thresholds : sequence of float
        Thresholds to evaluate.

    Returns
    -------
    dict of numpy.ndarray
        ``threshold``, ``micro_precision``, ``micro_recall``, ``micro_f1``,
        ``macro_f1``, ``macro_recall``, ``tray_fail_recall`` (fraction of defective
        trays caught) and ``tray_false_fail_rate`` (fraction of good trays rejected;
        NaN if there are none).
    """
    t = np.asarray(y_true).astype(bool)
    s = np.asarray(scores, dtype=float)
    out = {k: [] for k in ("threshold", "micro_precision", "micro_recall", "micro_f1",
                           "macro_f1", "macro_recall", "tray_fail_recall",
                           "tray_false_fail_rate")}
    n = t.shape[1]
    for thr in thresholds:
        p = s >= thr
        rep = multilabel_report(t, p, [str(i) for i in range(n)])
        tray = rep["tray"]
        out["threshold"].append(thr)
        out["micro_precision"].append(rep["micro"]["precision"])
        out["micro_recall"].append(rep["micro"]["recall"])
        out["micro_f1"].append(rep["micro"]["f1"])
        out["macro_f1"].append(rep["macro_f1"])
        out["macro_recall"].append(rep["macro_recall"])
        out["tray_fail_recall"].append(tray["fail_recall"])
        out["tray_false_fail_rate"].append(
            1 - tray["pass_recall"] if tray["n_pass_trays"] else float("nan"))
    return {k: np.asarray(v, dtype=float) for k, v in out.items()}


def best_threshold(sweep: Dict[str, np.ndarray], key: str = "macro_f1") -> float:
    """Threshold maximising a metric of a `threshold_sweep` result.

    Use this on the **validation** split only, then apply the chosen threshold to the
    test split, so the test result is not tuned on test data.

    Parameters
    ----------
    sweep : dict
        Output of `threshold_sweep`.
    key : str
        Metric to maximise. Default ``"macro_f1"``.

    Returns
    -------
    float
    """
    return float(sweep["threshold"][int(np.nanargmax(sweep[key]))])
