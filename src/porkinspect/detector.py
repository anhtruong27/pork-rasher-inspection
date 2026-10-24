"""Main model: a YOLO object detector fine-tuned on the pork rasher dataset.

Wraps the Ultralytics API so the rest of the project only sees plain NumPy arrays.
A detector outputs a bounding box, class and confidence for every defect it finds, so
unlike the baseline CNN it also tells an operator *where* the defect is.

YOLO ("You Only Look Once") is a single-stage detector: one forward pass of a CNN
predicts box coordinates and class scores on a grid of locations, followed by
non-maximum suppression (NMS) to remove duplicate boxes. We start from weights
pre-trained on COCO (transfer learning) because the dataset is small (~2k originals).

Ultralytics is imported lazily so the rest of the package (and the tests/docs) work
without it installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .decision import scores_from_detections


def train_detector(data_yaml: Union[str, Path], model_name: str = "yolov8n.pt",
                   epochs: int = 40, imgsz: int = 640, batch: int = 16,
                   project: str = "runs/detect", name: str = "yolo",
                   seed: int = 0, patience: int = 15, **kwargs):
    """Fine-tune a YOLO detector.

    Parameters
    ----------
    data_yaml : str or pathlib.Path
        Path to the dataset's ``data.yaml`` (Roboflow YOLOv8 export).
    model_name : str
        Starting weights, e.g. ``"yolov8n.pt"`` (nano, fastest) or ``"yolov8s.pt"``.
    epochs : int
        Maximum training epochs.
    imgsz : int
        Training/inference image size (longest side, pixels).
    batch : int
        Batch size.
    project, name : str
        Output folder ``<project>/<name>`` for weights, curves and logs.
    seed : int
        Random seed.
    patience : int
        Early-stopping patience (epochs without validation mAP improvement).
    **kwargs
        Passed to ``ultralytics.YOLO.train``.

    Returns
    -------
    ultralytics.YOLO
        Model loaded with the best checkpoint (selected on the validation split).
    """
    from ultralytics import YOLO

    # Newer Ultralytics versions put a RELATIVE project folder inside their own runs
    # folder (runs/detect/runs/detect/...), so pass an absolute path.
    project = str(Path(project).resolve())
    model = YOLO(model_name)
    model.train(data=str(data_yaml), epochs=epochs, imgsz=imgsz, batch=batch,
                project=project, name=name, seed=seed, patience=patience,
                deterministic=True, exist_ok=True, **kwargs)
    trainer = getattr(model, "trainer", None)
    save_dir = Path(getattr(trainer, "save_dir", Path(project) / name))
    best = save_dir / "weights" / "best.pt"
    return YOLO(str(best)) if best.exists() else model


def box_metrics(model, data_yaml: Union[str, Path], split: str = "test",
                imgsz: int = 640, workers: int = 2) -> Dict[str, object]:
    """Box-level detection metrics from the Ultralytics validator.

    Parameters
    ----------
    model : ultralytics.YOLO
        Trained detector.
    data_yaml : str or pathlib.Path
        Dataset config.
    split : str
        ``"val"`` or ``"test"``.
    imgsz : int
        Image size.
    workers : int
        Image-loading processes (Ultralytics' default of 8 can exhaust RAM on a laptop).

    Returns
    -------
    dict
        ``map50``, ``map50_95``, ``precision``, ``recall`` (means over classes) and
        ``per_class_map50`` (dict).
    """
    r = model.val(data=str(data_yaml), split=split, imgsz=imgsz, verbose=False, plots=True,
                  workers=workers)
    names = r.names if hasattr(r, "names") else model.names
    per_class = {}
    for i, c in enumerate(r.box.ap_class_index):
        per_class[names[int(c)]] = float(r.box.ap50[i])
    return {"map50": float(r.box.map50), "map50_95": float(r.box.map),
            "precision": float(r.box.mp), "recall": float(r.box.mr),
            "per_class_map50": per_class}


def predict_detections(model, images: Sequence[Union[str, Path, np.ndarray]],
                       conf: float = 0.01, imgsz: int = 640, batch: int = 16
                       ) -> List[List[Tuple[int, float, Tuple[float, float, float, float]]]]:
    """Run the detector and return plain-Python detections.

    Parameters
    ----------
    model : ultralytics.YOLO
        Trained detector.
    images : sequence
        Image paths or RGB arrays.
    conf : float
        Minimum confidence kept. Use a low value (default 0.01) so that the decision
        threshold can be swept later without re-running the model.
    imgsz : int
        Inference size.
    batch : int
        Images per forward pass.

    Returns
    -------
    list
        One list per image of ``(class_id, confidence, (x1, y1, x2, y2))`` in pixels.
    """
    out = []
    for i in range(0, len(images), batch):
        chunk = [str(p) if isinstance(p, Path) else p for p in images[i:i + batch]]
        for r in model.predict(chunk, conf=conf, imgsz=imgsz, verbose=False):
            b = r.boxes
            dets = [(int(c), float(s), tuple(float(v) for v in xyxy))
                    for c, s, xyxy in zip(b.cls.cpu().numpy(), b.conf.cpu().numpy(),
                                          b.xyxy.cpu().numpy())]
            out.append(dets)
    return out


def detections_to_scores(detections: Sequence[Sequence[Tuple]], n_classes: int) -> np.ndarray:
    """Convert detector output to the (n_images, n_classes) presence-score matrix.

    Parameters
    ----------
    detections : sequence
        Output of `predict_detections`.
    n_classes : int
        Number of classes.

    Returns
    -------
    numpy.ndarray
        Shape (n_images, n_classes).
    """
    return np.stack([scores_from_detections([(d[0], d[1]) for d in dets], n_classes)
                     for dets in detections]) if len(detections) else np.zeros((0, n_classes))


def model_gflops(model, imgsz: int = 640) -> Optional[float]:
    """GFLOPs for one image at ``imgsz``.

    Tries Ultralytics' own counter first; if that is unavailable (it returned nothing in
    run 1), falls back to `porkinspect.performance.count_flops` on the underlying
    PyTorch network, which counts Conv2d and Linear layers only.

    Parameters
    ----------
    model : ultralytics.YOLO
        Detector.
    imgsz : int
        Square input size in pixels.

    Returns
    -------
    float or None
        GFLOPs per image, or None if neither method works.
    """
    try:
        from ultralytics.utils.torch_utils import get_flops

        g = float(get_flops(model.model, imgsz))
        if g > 0:
            return g
    except Exception:  # pragma: no cover - depends on ultralytics version
        pass
    try:
        import copy

        import torch

        from .performance import count_flops

        net = copy.deepcopy(model.model).float().cpu()
        return count_flops(net, (1, 3, imgsz, imgsz))["flops"] / 1e9
    except Exception:  # pragma: no cover
        return None
