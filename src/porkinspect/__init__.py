"""porkinspect: automated visual inspection of packaged pork rasher trays.

MMA3001 (Numerical Methods and Machine Learning) individual project, Monash University, 2026.

Engineering problem
-------------------
Manual visual inspection of packaged pork rashers is slow, tiring and inconsistent.
This package takes a single top-down RGB image of a tray and

1. detects and localises packaging/product defects (bounding box + class + confidence),
2. converts the detections into a tray-level decision (PASS / FAIL, and which defect
   types are present), and
3. provides the tools needed to validate that decision and judge whether it is fast
   enough for a production line.

Defect classes (from the Roboflow dataset): ``loose-meat``, ``packaging-error``,
``twisted-meat``, ``unsealed``, ``wrinkle``.

Modules
-------
- `porkinspect.io_utils`     – input validation (format, size, channels, dtype).
- `porkinspect.data`         – YOLO label parsing, dataset audit, leakage checks.
- `porkinspect.decision`     – detections -> per-class presence -> PASS/FAIL.
- `porkinspect.metrics`      – multi-label metrics, balanced accuracy, threshold sweeps.
- `porkinspect.baseline`     – from-scratch CNN tray classifier (baseline model).
- `porkinspect.detector`     – wrapper around an Ultralytics YOLO detector (main model).
- `porkinspect.performance`  – timing, FLOP counting and arithmetic-intensity estimates.
- `porkinspect.occlusion`    – occlusion-sensitivity maps for error investigation.
"""

__version__ = "1.0.0"

#: Default class order used by the Roboflow export (alphabetical). The order actually
#: used is always read from the dataset's ``data.yaml`` where possible.
DEFAULT_CLASS_NAMES = (
    "loose-meat",
    "packaging-error",
    "twisted-meat",
    "unsealed",
    "wrinkle",
)
