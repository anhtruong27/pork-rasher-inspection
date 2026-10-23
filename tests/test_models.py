"""Tests for the baseline CNN, FLOP counting, timing and occlusion maps."""
import numpy as np
import pytest
import torch
import torch.nn as nn

from porkinspect.baseline import TrayCNN, TrayDataset, positive_weights, preprocess
from porkinspect.data import audit_dataset
from porkinspect.occlusion import occlusion_map
from porkinspect.performance import (arithmetic_intensity, count_flops,
                                     required_throughput, time_function)


def test_cnn_output_shape():
    m = TrayCNN(5).eval()
    assert m(torch.zeros(2, 3, 240, 320)).shape == (2, 5)


def test_preprocess_shape_and_normalisation():
    x = preprocess(np.full((540, 720, 3), 124, np.uint8))
    assert x.shape == (3, 240, 320)
    assert abs(float(x.mean())) < 0.2         # ~ImageNet mean grey -> ~0


def test_dataset_items(tiny_dataset):
    root, names = tiny_dataset
    _, rec = audit_dataset(root)
    ds = TrayDataset(rec["train"], 5)
    x, y = ds[0]
    assert x.shape == (3, 240, 320) and y.tolist() == [1, 0, 0, 1, 0]


def test_positive_weights(tiny_dataset):
    root, _ = tiny_dataset
    _, rec = audit_dataset(root)
    w = positive_weights(rec["train"], 5)
    # unsealed in 2/3 trays -> neg/pos = 0.5 -> clipped to 1; loose-meat 1/3 -> 2
    assert w[3] == pytest.approx(1.0) and w[0] == pytest.approx(2.0)


def test_flops_single_conv_matches_hand_calculation():
    m = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1, bias=False))
    r = count_flops(m, (1, 3, 10, 10))
    # MACs = C_out*H*W*C_in*k*k = 8*10*10*3*9 = 21600 -> FLOPs = 43200
    assert r["flops"] == 43200
    assert r["bytes"] == 4 * (3 * 100 + 8 * 100 + 8 * 3 * 9)


def test_flops_linear():
    r = count_flops(nn.Sequential(nn.Linear(10, 4)), (1, 10))
    assert r["flops"] == 80


def test_arithmetic_intensity_and_errors():
    assert arithmetic_intensity(100, 50) == 2
    with pytest.raises(ValueError):
        arithmetic_intensity(1, 0)


def test_required_throughput():
    r = required_throughput(120)
    assert r["required_fps"] == 2 and r["budget_ms"] == 500
    with pytest.raises(ValueError):
        required_throughput(0)


def test_time_function_returns_stats():
    r = time_function(lambda: sum(range(1000)), warmup=1, repeats=5)
    assert r["min_ms"] <= r["median_ms"] <= r["max_ms"]
    with pytest.raises(ValueError):
        time_function(lambda: None, repeats=0)


def test_occlusion_finds_the_important_region():
    img = np.zeros((120, 160, 3), np.uint8)
    img[30:60, 100:140] = 255                  # the "defect"
    score = lambda im: im[30:60, 100:140].mean() / 255.0
    heat, base = occlusion_map(score, img, patch=20, stride=10)
    assert base == pytest.approx(1.0)
    y, x = np.unravel_index(np.argmax(heat), heat.shape)
    assert 30 <= y < 60 and 100 <= x < 140
    assert heat[:20, :20].max() == 0           # covering background changes nothing


def test_occlusion_invalid_args():
    with pytest.raises(ValueError):
        occlusion_map(lambda im: 0.0, np.zeros((50, 50, 3), np.uint8), patch=0)
    with pytest.raises(ValueError):
        occlusion_map(lambda im: 0.0, np.zeros((50, 50, 3), np.uint8), patch=60)


def test_overlay_colour_scale_starts_at_zero():
    import matplotlib
    matplotlib.use("Agg")
    from porkinspect.occlusion import overlay
    img = np.zeros((60, 80, 3), np.uint8)
    ax = overlay(img, np.full((60, 80), -0.01))        # all tiny negatives (run-1 bug case)
    lo, hi = ax.images[-1].get_clim()
    assert lo == 0.0 and hi == pytest.approx(0.05)
