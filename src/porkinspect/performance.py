"""Timing, FLOP counting and arithmetic-intensity estimates.

Used for the optimisation/performance section: is the method fast enough for a
production line, and *why* does one model cost more than another?

- `time_function` measures wall-clock latency with warm-up runs and repeats, and
  reports median and spread (single timings are unreliable).
- `count_flops` counts multiply-add operations of Conv2d and Linear layers with
  forward hooks (1 multiply-add = 2 FLOPs).
- `arithmetic_intensity` = FLOPs / bytes moved. Low intensity means memory-bound,
  high intensity means compute-bound (roofline model, MMA3001 Week 2).
- `required_throughput` converts a line speed into a time budget per tray.
"""

from __future__ import annotations

import statistics
import time
from typing import Callable, Dict, Tuple

import torch
import torch.nn as nn


def time_function(fn: Callable[[], object], warmup: int = 3, repeats: int = 20,
                  sync_cuda: bool = True) -> Dict[str, float]:
    """Time a zero-argument callable.

    Parameters
    ----------
    fn : callable
        Function to time (e.g. ``lambda: model(x)``).
    warmup : int
        Untimed calls first (caches, GPU kernels, lazy initialisation).
    repeats : int
        Timed calls.
    sync_cuda : bool
        Call ``torch.cuda.synchronize()`` around each call when CUDA is available,
        otherwise asynchronous GPU work would be under-counted.

    Returns
    -------
    dict
        ``median_ms``, ``mean_ms``, ``std_ms``, ``min_ms``, ``max_ms``.
    """
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    sync = sync_cuda and torch.cuda.is_available()
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        if sync:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if sync:
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    return {
        "median_ms": statistics.median(times),
        "mean_ms": statistics.fmean(times),
        "std_ms": statistics.pstdev(times),
        "min_ms": min(times),
        "max_ms": max(times),
    }


def count_flops(model: nn.Module, input_shape: Tuple[int, ...]) -> Dict[str, float]:
    """Count FLOPs and memory traffic of Conv2d and Linear layers for one forward pass.

    Parameters
    ----------
    model : torch.nn.Module
        Network.
    input_shape : tuple of int
        Input shape including batch, e.g. ``(1, 3, 240, 320)``.

    Returns
    -------
    dict
        ``flops`` (2 x multiply-adds), ``params`` (count), ``bytes`` (float32 weights
        + input and output activations of counted layers) and ``arithmetic_intensity``
        (FLOPs/byte).

    Notes
    -----
    For a convolution with C_in input channels, C_out output channels, k x k kernel
    and an H_out x W_out output: MACs = C_out * H_out * W_out * C_in * k * k / groups.
    BatchNorm, ReLU and pooling are ignored (negligible compared with convolutions).
    """
    stats = {"macs": 0, "bytes": 0}

    def conv_hook(m: nn.Conv2d, inp, out):
        k = m.kernel_size[0] * m.kernel_size[1]
        macs = out.numel() * (m.in_channels // m.groups) * k
        stats["macs"] += macs
        stats["bytes"] += 4 * (inp[0].numel() + out.numel() + m.weight.numel())

    def lin_hook(m: nn.Linear, inp, out):
        stats["macs"] += out.numel() * m.in_features
        stats["bytes"] += 4 * (inp[0].numel() + out.numel() + m.weight.numel())

    hooks = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(lin_hook))
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            p = next(model.parameters())
            model(torch.zeros(input_shape, device=p.device, dtype=p.dtype))
    finally:
        for h in hooks:
            h.remove()
        model.train(was_training)
    flops = 2.0 * stats["macs"]
    params = sum(p.numel() for p in model.parameters())
    return {"flops": flops, "params": float(params), "bytes": float(stats["bytes"]),
            "arithmetic_intensity": arithmetic_intensity(flops, stats["bytes"])}


def arithmetic_intensity(flops: float, bytes_moved: float) -> float:
    """FLOPs per byte of memory traffic.

    Parameters
    ----------
    flops : float
        Floating-point operations.
    bytes_moved : float
        Bytes read + written.

    Returns
    -------
    float
    """
    if bytes_moved <= 0:
        raise ValueError("bytes_moved must be positive")
    return flops / bytes_moved


def required_throughput(trays_per_minute: float) -> Dict[str, float]:
    """Time budget per tray for a given line speed.

    Parameters
    ----------
    trays_per_minute : float
        Packaging line speed.

    Returns
    -------
    dict
        ``budget_ms`` per tray and ``required_fps``.
    """
    if trays_per_minute <= 0:
        raise ValueError("trays_per_minute must be positive")
    fps = trays_per_minute / 60.0
    return {"budget_ms": 1000.0 / fps, "required_fps": fps}
