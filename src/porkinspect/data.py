"""Dataset loading and auditing for the Roboflow YOLO export.

Expected folder layout (Roboflow "YOLOv8" export)::

    <root>/
        data.yaml
        train/images/*.jpg   train/labels/*.txt
        valid/images/*.jpg   valid/labels/*.txt
        test/images/*.jpg    test/labels/*.txt

Each label file has one line per box: ``class_id x_center y_center width height``,
with coordinates normalised to [0, 1] by the image width/height. An empty or missing
label file means the image has no annotated defect.

The audit answers questions that must be settled *before* training:

- How many images and boxes per split and per class (class imbalance)?
- Are there defect-free trays (needed for a PASS class)?
- Are there malformed labels?
- Do augmented copies of the same original photo appear in more than one split
  (data leakage that would inflate test scores)?
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

SPLITS = ("train", "valid", "test")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# Roboflow renames files to "<original>_<ext>.rf.<hash>.<ext>"; augmented copies of the
# same photo share the "<original>" part.
_RF_PATTERN = re.compile(r"^(?P<stem>.+?)(?:_(?:jpg|jpeg|png|bmp))?\.rf\.[0-9a-fA-F]+$")


class LabelFormatError(ValueError):
    """Raised when a YOLO label line cannot be parsed or is out of range."""


@dataclass
class ImageRecord:
    """One image and its boxes.

    Attributes
    ----------
    image_path : pathlib.Path
        Path to the image file.
    boxes : numpy.ndarray
        Array of shape (n, 5): ``class_id, xc, yc, w, h`` (normalised). ``n`` may be 0.
    split : str
        ``"train"``, ``"valid"`` or ``"test"``.
    """

    image_path: Path
    boxes: np.ndarray
    split: str

    @property
    def source_id(self) -> str:
        """Identifier of the original photo this image was derived from."""
        return source_stem(self.image_path.name)


@dataclass
class AuditReport:
    """Summary statistics returned by `audit_dataset`."""

    class_names: List[str]
    images_per_split: Dict[str, int] = field(default_factory=dict)
    empty_images_per_split: Dict[str, int] = field(default_factory=dict)
    boxes_per_class: Dict[str, Dict[str, int]] = field(default_factory=dict)
    images_with_class: Dict[str, Dict[str, int]] = field(default_factory=dict)
    malformed_files: List[str] = field(default_factory=list)
    leaked_sources: Dict[str, List[str]] = field(default_factory=dict)

    def to_text(self) -> str:
        """Return a human-readable multi-line summary."""
        lines = ["DATASET AUDIT", "============="]
        for s in self.images_per_split:
            lines.append(
                f"{s:>5}: {self.images_per_split[s]:5d} images, "
                f"{self.empty_images_per_split.get(s, 0):4d} with no defect boxes"
            )
        lines.append("")
        header = "class".ljust(18) + "".join(f"{s:>16}" for s in self.images_per_split)
        lines.append(header + "   (boxes / images containing class)")
        for c in self.class_names:
            row = c.ljust(18)
            for s in self.images_per_split:
                row += f"{self.boxes_per_class[s].get(c, 0):>9d} / {self.images_with_class[s].get(c, 0):<5d}"
            lines.append(row)
        lines.append("")
        lines.append(f"Malformed label files: {len(self.malformed_files)}")
        lines.append(
            f"Original photos appearing in more than one split (leakage): "
            f"{len(self.leaked_sources)}"
        )
        for src, splits in list(self.leaked_sources.items())[:10]:
            lines.append(f"   {src}: {splits}")
        return "\n".join(lines)


def source_stem(filename: str) -> str:
    """Recover the original-photo identifier from a Roboflow file name.

    Parameters
    ----------
    filename : str
        e.g. ``"IMG_0042_jpg.rf.3f2a9c.jpg"``.

    Returns
    -------
    str
        e.g. ``"IMG_0042"``. Names without the Roboflow suffix are returned without
        their extension.

    Examples
    --------
    >>> source_stem("IMG_0042_jpg.rf.3f2a9c.jpg")
    'IMG_0042'
    >>> source_stem("tray7.png")
    'tray7'
    """
    stem = Path(filename).stem
    m = _RF_PATTERN.match(stem)
    return m.group("stem") if m else stem


def parse_yolo_label(path: Path, n_classes: Optional[int] = None) -> np.ndarray:
    """Parse one YOLO label file.

    Parameters
    ----------
    path : pathlib.Path
        Label file. A missing file is treated as "no boxes".
    n_classes : int, optional
        If given, class ids must lie in ``[0, n_classes)``.

    Returns
    -------
    numpy.ndarray
        Shape (n, 5), dtype float: ``class_id, xc, yc, w, h``.

    Raises
    ------
    LabelFormatError
        If a line does not have 5 numbers, a class id is invalid, or a coordinate is
        outside [0, 1] (a tolerance of 1e-3 is allowed for rounding), or a box has
        non-positive size.
    """
    path = Path(path)
    if not path.exists():
        return np.zeros((0, 5), dtype=float)
    rows = []
    for i, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            # Polygon (segmentation) labels have more than 5 values; reduce them to a box.
            if len(parts) > 5 and (len(parts) - 1) % 2 == 0:
                cls = parts[0]
                xy = np.asarray(parts[1:], dtype=float).reshape(-1, 2)
                x0, y0 = xy.min(axis=0)
                x1, y1 = xy.max(axis=0)
                parts = [cls, (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0]
            else:
                raise LabelFormatError(f"{path.name}:{i}: expected 5 values, got {len(parts)}")
        try:
            cls = float(parts[0])
            coords = [float(v) for v in parts[1:]]
        except ValueError as exc:
            raise LabelFormatError(f"{path.name}:{i}: non-numeric value") from exc
        if cls != int(cls) or cls < 0 or (n_classes is not None and cls >= n_classes):
            raise LabelFormatError(f"{path.name}:{i}: invalid class id {parts[0]}")
        tol = 1e-3
        if any(v < -tol or v > 1 + tol for v in coords):
            raise LabelFormatError(f"{path.name}:{i}: coordinate outside [0, 1]")
        if coords[2] <= 0 or coords[3] <= 0:
            raise LabelFormatError(f"{path.name}:{i}: box has non-positive size")
        rows.append([cls] + coords)
    return np.asarray(rows, dtype=float).reshape(-1, 5)


def read_class_names(root: Path) -> List[str]:
    """Read class names from ``<root>/data.yaml``.

    Parameters
    ----------
    root : pathlib.Path
        Dataset root folder.

    Returns
    -------
    list of str
        Class names in label-id order.
    """
    import yaml  # local import keeps the dependency optional for docs/tests

    cfg = yaml.safe_load((Path(root) / "data.yaml").read_text())
    names = cfg["names"]
    if isinstance(names, dict):  # newer Ultralytics format {0: 'a', 1: 'b'}
        names = [names[k] for k in sorted(names)]
    return list(names)


def scan_split(root: Path, split: str, n_classes: Optional[int] = None,
               errors: Optional[List[str]] = None) -> List[ImageRecord]:
    """List all images of a split with their boxes.

    Parameters
    ----------
    root : pathlib.Path
        Dataset root.
    split : str
        ``"train"``, ``"valid"`` or ``"test"``.
    n_classes : int, optional
        Number of classes, used to validate class ids.
    errors : list of str, optional
        If given, malformed label files are appended here and skipped instead of
        raising.

    Returns
    -------
    list of ImageRecord
    """
    img_dir = Path(root) / split / "images"
    lbl_dir = Path(root) / split / "labels"
    records = []
    if not img_dir.exists():
        return records
    for img in sorted(img_dir.iterdir()):
        if img.suffix.lower() not in IMAGE_EXTS:
            continue
        lbl = lbl_dir / (img.stem + ".txt")
        try:
            boxes = parse_yolo_label(lbl, n_classes)
        except LabelFormatError as exc:
            if errors is None:
                raise
            errors.append(str(exc))
            continue
        records.append(ImageRecord(img, boxes, split))
    return records


def presence_vector(boxes: np.ndarray, n_classes: int) -> np.ndarray:
    """Convert boxes to a multi-hot "which defect types are present" vector.

    Parameters
    ----------
    boxes : numpy.ndarray
        Shape (n, 5) with class ids in column 0.
    n_classes : int
        Number of defect classes.

    Returns
    -------
    numpy.ndarray
        Shape (n_classes,), dtype ``float32``, 1.0 where the class appears.
    """
    v = np.zeros(n_classes, dtype=np.float32)
    if len(boxes):
        v[np.unique(boxes[:, 0].astype(int))] = 1.0
    return v


def find_leakage(records: Sequence[ImageRecord]) -> Dict[str, List[str]]:
    """Find original photos whose derived images appear in more than one split.

    Parameters
    ----------
    records : sequence of ImageRecord
        Records from all splits.

    Returns
    -------
    dict
        ``{source_id: sorted list of splits}`` for every leaked source.
    """
    splits_by_source: Dict[str, set] = defaultdict(set)
    for r in records:
        splits_by_source[r.source_id].add(r.split)
    return {s: sorted(v) for s, v in splits_by_source.items() if len(v) > 1}


def audit_dataset(root: Path, class_names: Optional[Sequence[str]] = None
                  ) -> Tuple[AuditReport, Dict[str, List[ImageRecord]]]:
    """Audit a YOLO-format dataset.

    Parameters
    ----------
    root : pathlib.Path
        Dataset root containing ``data.yaml`` and split folders.
    class_names : sequence of str, optional
        Overrides the names in ``data.yaml``.

    Returns
    -------
    report : AuditReport
        Counts, empty images, malformed files and leakage.
    records : dict
        ``{split: list of ImageRecord}``.
    """
    root = Path(root)
    names = list(class_names) if class_names else read_class_names(root)
    report = AuditReport(class_names=names)
    records: Dict[str, List[ImageRecord]] = {}
    for split in SPLITS:
        recs = scan_split(root, split, len(names), report.malformed_files)
        if not recs and not (root / split).exists():
            continue
        records[split] = recs
        report.images_per_split[split] = len(recs)
        report.empty_images_per_split[split] = sum(len(r.boxes) == 0 for r in recs)
        box_counts: Counter = Counter()
        img_counts: Counter = Counter()
        for r in recs:
            ids = r.boxes[:, 0].astype(int)
            box_counts.update(names[i] for i in ids)
            img_counts.update(names[i] for i in np.unique(ids))
        report.boxes_per_class[split] = dict(box_counts)
        report.images_with_class[split] = dict(img_counts)
    report.leaked_sources = find_leakage([r for v in records.values() for r in v])
    return report, records


_FRAME_PATTERN = re.compile(r"^(?P<prefix>.*?)(?P<num>\d+)$")


def frame_key(source_id: str) -> Tuple[str, Optional[int]]:
    """Split a source id into a sequence prefix and a frame number.

    Many images in this dataset are frames from a video (``frame_with_red_7506``).
    Neighbouring frames are nearly identical, so they must be treated as duplicates
    when checking for leakage.

    Parameters
    ----------
    source_id : str
        Output of `source_stem`.

    Returns
    -------
    prefix : str
        Text before the trailing number (``"frame_with_red_"``).
    number : int or None
        Trailing frame number, or None if the id does not end in digits.

    Examples
    --------
    >>> frame_key("frame_with_red_7506")
    ('frame_with_red_', 7506)
    >>> frame_key("tray")
    ('tray', None)
    """
    m = _FRAME_PATTERN.match(source_id)
    if not m:
        return source_id, None
    return m.group("prefix"), int(m.group("num"))


def near_duplicate_leaks(train: Sequence[ImageRecord], held_out: Sequence[ImageRecord],
                         window: int = 30) -> List[int]:
    """Indices of held-out images that have a near-identical frame in the training set.

    A held-out image is flagged if a training image has the same sequence prefix and
    a frame number within ``window`` frames (``window=0`` reduces to the exact
    same-photo check of `find_leakage`).

    Parameters
    ----------
    train : sequence of ImageRecord
        Training records.
    held_out : sequence of ImageRecord
        Validation or test records.
    window : int
        Maximum frame distance treated as a near-duplicate. At ~30 frames/s,
        ``window=30`` is about one second of video.

    Returns
    -------
    list of int
        Indices into ``held_out`` of leaked images.

    Raises
    ------
    ValueError
        If ``window`` is negative.
    """
    if window < 0:
        raise ValueError("window must be >= 0")
    frames: Dict[str, np.ndarray] = {}
    exact = set()
    tmp: Dict[str, List[int]] = defaultdict(list)
    for r in train:
        prefix, num = frame_key(r.source_id)
        exact.add(r.source_id)
        if num is not None:
            tmp[prefix].append(num)
    frames = {k: np.sort(np.asarray(v)) for k, v in tmp.items()}
    leaked = []
    for i, r in enumerate(held_out):
        prefix, num = frame_key(r.source_id)
        if r.source_id in exact:
            leaked.append(i)
            continue
        if num is None or prefix not in frames:
            continue
        arr = frames[prefix]
        j = np.searchsorted(arr, num)
        nearest = min(abs(arr[k] - num) for k in (j - 1, j) if 0 <= k < len(arr))
        if nearest <= window:
            leaked.append(i)
    return leaked


def unique_sources(records: Sequence[ImageRecord]) -> Dict[str, List[ImageRecord]]:
    """Group records by original photo (augmented copies share a source).

    Parameters
    ----------
    records : sequence of ImageRecord

    Returns
    -------
    dict
        ``{source_id: [records]}``, records sorted by file name.
    """
    groups: Dict[str, List[ImageRecord]] = defaultdict(list)
    for r in records:
        groups[r.source_id].append(r)
    return {k: sorted(v, key=lambda r: r.image_path.name) for k, v in groups.items()}


def frame_gaps(records: Sequence[ImageRecord]) -> np.ndarray:
    """Differences between consecutive frame numbers of the original photos.

    Used to choose the segment gap ``G`` for `segment_split`: small gaps are frames of
    one continuous shot, large gaps are breaks between shots.

    Parameters
    ----------
    records : sequence of ImageRecord
        Records from all splits.

    Returns
    -------
    numpy.ndarray
        Gaps (int) between neighbouring frame numbers within each sequence prefix.
    """
    by_prefix: Dict[str, set] = defaultdict(set)
    for src in unique_sources(records):
        prefix, num = frame_key(src)
        if num is not None:
            by_prefix[prefix].add(num)
    gaps = [np.diff(sorted(v)) for v in by_prefix.values() if len(v) > 1]
    return np.concatenate(gaps).astype(int) if gaps else np.zeros(0, dtype=int)


def assign_segments(records: Sequence[ImageRecord], gap: int) -> Dict[str, str]:
    """Label every original photo with a segment id.

    Photos of the same prefix are sorted by frame number; a new segment starts wherever
    the jump to the next frame number is larger than ``gap``. Photos without a frame
    number form their own one-photo segment.

    Parameters
    ----------
    records : sequence of ImageRecord
    gap : int
        Largest frame jump that still counts as the same segment (``G``).

    Returns
    -------
    dict
        ``{source_id: segment_id}``.

    Raises
    ------
    ValueError
        If ``gap`` is negative.
    """
    if gap < 0:
        raise ValueError("gap must be >= 0")
    by_prefix: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
    seg: Dict[str, str] = {}
    for src in unique_sources(records):
        prefix, num = frame_key(src)
        if num is None:
            seg[src] = f"single:{src}"
        else:
            by_prefix[prefix].append((num, src))
    for prefix, items in by_prefix.items():
        items.sort()
        k, prev = 0, None
        for num, src in items:
            if prev is not None and num - prev > gap:
                k += 1
            seg[src] = f"{prefix}#{k}"
            prev = num
    return seg


def segment_split(records: Sequence[ImageRecord], gap: int, test_frac: float = 0.15,
                  valid_frac: float = 0.10, n_classes: int = 5,
                  must_have: Sequence[int] = (), min_pass: int = 1,
                  seed: int = 0, max_tries: int = 200) -> Dict[str, List[ImageRecord]]:
    """Leak-free split: whole segments go to train, valid or test.

    Every augmented copy of a photo follows its photo, and every photo follows its
    segment, so no test photo has a training photo within ``gap`` frames of it.
    Valid and test keep ONE copy per original photo (augmentation is for training only).

    Parameters
    ----------
    records : sequence of ImageRecord
        All records from the original export (any split).
    gap : int
        Segment gap ``G`` (see `assign_segments`).
    test_frac, valid_frac : float
        Target share of ORIGINAL photos in test and valid.
    n_classes : int
        Number of classes.
    must_have : sequence of int
        Class ids that must appear in BOTH valid and test (e.g. the two main classes).
    min_pass : int
        Minimum number of defect-free trays in both valid and test.
    seed : int
        Random seed; the first seed (seed, seed+1, ...) meeting the requirements is used.
    max_tries : int
        Number of seeds to try.

    Returns
    -------
    dict
        ``{"train": [...], "valid": [...], "test": [...], "seed": int}`` (records keep
        their file paths; their ``split`` attribute is updated).

    Raises
    ------
    RuntimeError
        If no seed meets the requirements (try a smaller ``gap`` or other fractions).
    """
    groups = unique_sources(records)
    seg_of = assign_segments(records, gap)
    segments: Dict[str, List[str]] = defaultdict(list)
    for src, s in seg_of.items():
        segments[s].append(src)
    names = sorted(segments)
    n_orig = len(groups)

    def ok(srcs: List[str]) -> bool:
        pres = np.zeros(n_classes)
        n_pass = 0
        for src in srcs:
            boxes = groups[src][0].boxes
            pres += presence_vector(boxes, n_classes)
            n_pass += int(len(boxes) == 0)
        return n_pass >= min_pass and all(pres[c] > 0 for c in must_have)

    for k in range(max_tries):
        rng = np.random.default_rng(seed + k)
        order = [names[i] for i in rng.permutation(len(names))]
        test, valid, train = [], [], []
        for s in order:
            if len(test) < test_frac * n_orig:
                test += segments[s]
            elif len(valid) < valid_frac * n_orig:
                valid += segments[s]
            else:
                train += segments[s]
        if ok(test) and ok(valid):
            out: Dict[str, object] = {"seed": seed + k}
            for split, srcs in (("train", train), ("valid", valid), ("test", test)):
                recs = []
                for src in sorted(srcs):
                    copies = groups[src] if split == "train" else groups[src][:1]
                    recs += [ImageRecord(r.image_path, r.boxes, split) for r in copies]
                out[split] = recs
            return out  # type: ignore[return-value]
    raise RuntimeError("no split met the requirements; try a smaller gap")


def write_yolo_split(split: Dict[str, List[ImageRecord]], out_root: Path,
                     class_names: Sequence[str]) -> Path:
    """Write a split as a YOLO dataset folder (copied images + label files).

    Parameters
    ----------
    split : dict
        Output of `segment_split`.
    out_root : pathlib.Path
        Folder to create.
    class_names : sequence of str
        Names in class-id order.

    Returns
    -------
    pathlib.Path
        Path of the new ``data.yaml``.
    """
    import shutil

    import yaml

    out_root = Path(out_root).resolve()
    for name in ("train", "valid", "test"):
        img_dir, lbl_dir = out_root / name / "images", out_root / name / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for r in split[name]:
            dst = img_dir / r.image_path.name
            if not dst.exists():
                shutil.copy2(r.image_path, dst)
            lines = [f"{int(b[0])} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}" for b in r.boxes]
            (lbl_dir / (r.image_path.stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""))
    cfg = {"path": str(out_root), "train": "train/images", "val": "valid/images",
           "test": "test/images", "nc": len(class_names), "names": list(class_names)}
    yaml_path = out_root / "data.yaml"
    yaml_path.write_text(yaml.safe_dump(cfg))
    return yaml_path
