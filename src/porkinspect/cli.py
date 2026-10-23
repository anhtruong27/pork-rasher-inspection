"""Command-line tray inspection.

Usage::

    python -m porkinspect.cli path/to/tray.jpg --weights best.pt --threshold 0.35

Prints one line per image, e.g. ``tray.jpg  FAIL: unsealed (0.91)``, and optionally
saves an annotated copy of the image with ``--save-dir``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from . import DEFAULT_CLASS_NAMES
from .decision import decision_label, scores_from_detections
from .io_utils import InvalidImageError, load_image


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments (defaults to ``sys.argv[1:]``).

    Returns
    -------
    int
        Exit code: 0 if every image was processed, 1 if any input was invalid.
    """
    ap = argparse.ArgumentParser(description="Inspect pork rasher tray images.")
    ap.add_argument("images", nargs="+", help="image files")
    ap.add_argument("--weights", required=True, help="trained YOLO weights (best.pt)")
    ap.add_argument("--threshold", type=float, default=0.5, help="decision threshold")
    ap.add_argument("--save-dir", default=None, help="save annotated images here")
    args = ap.parse_args(argv)
    if not 0.0 <= args.threshold <= 1.0:
        ap.error("--threshold must be in [0, 1]")

    from ultralytics import YOLO

    model = YOLO(args.weights)
    names = [model.names[i] for i in sorted(model.names)] if model.names else list(DEFAULT_CLASS_NAMES)
    status = 0
    for path in args.images:
        try:
            image, resized = load_image(path)
        except (FileNotFoundError, InvalidImageError) as exc:
            print(f"{path}  ERROR: {exc}", file=sys.stderr)
            status = 1
            continue
        r = model.predict(image[:, :, ::-1].copy(), conf=0.01, verbose=False)[0]  # BGR for YOLO
        dets = list(zip(r.boxes.cls.cpu().numpy().astype(int), r.boxes.conf.cpu().numpy()))
        scores = scores_from_detections(dets, len(names))
        note = " (resized)" if resized else ""
        print(f"{Path(path).name}{note}  {decision_label(scores, names, args.threshold)}")
        if args.save_dir:
            from PIL import Image

            Path(args.save_dir).mkdir(parents=True, exist_ok=True)
            keep = r.boxes.conf.cpu().numpy() >= args.threshold
            r.boxes = r.boxes[keep]
            Image.fromarray(r.plot()[:, :, ::-1]).save(Path(args.save_dir) / Path(path).name)
    return status


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
