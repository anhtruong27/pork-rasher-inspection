"""Tests for label parsing and the dataset audit (porkinspect.data)."""
import numpy as np
import pytest

from porkinspect.data import (LabelFormatError, audit_dataset, parse_yolo_label,
                              presence_vector, source_stem)


def test_source_stem_roboflow_name():
    assert source_stem("IMG_0042_jpg.rf.3f2a9c.jpg") == "IMG_0042"


def test_source_stem_plain_name():
    assert source_stem("tray7.png") == "tray7"


def test_parse_valid_label(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("0 0.5 0.5 0.2 0.3\n4 0.1 0.2 0.05 0.05\n\n")
    b = parse_yolo_label(p, n_classes=5)
    assert b.shape == (2, 5) and b[1, 0] == 4


def test_missing_label_means_no_boxes(tmp_path):
    assert parse_yolo_label(tmp_path / "none.txt").shape == (0, 5)


def test_polygon_label_reduced_to_box(tmp_path):
    p = tmp_path / "poly.txt"
    p.write_text("2 0.1 0.1 0.3 0.1 0.3 0.5 0.1 0.5\n")
    b = parse_yolo_label(p, 5)
    np.testing.assert_allclose(b[0], [2, 0.2, 0.3, 0.2, 0.4])


@pytest.mark.parametrize("line", [
    "0 0.5 0.5 0.2",          # too few values
    "7 0.5 0.5 0.2 0.2",      # class out of range
    "0 1.5 0.5 0.2 0.2",      # coordinate out of range
    "0 0.5 0.5 0.0 0.2",      # zero width
    "a 0.5 0.5 0.2 0.2",      # non-numeric
    "1.5 0.5 0.5 0.2 0.2",    # non-integer class
])
def test_malformed_labels_rejected(tmp_path, line):
    p = tmp_path / "bad.txt"
    p.write_text(line + "\n")
    with pytest.raises(LabelFormatError):
        parse_yolo_label(p, n_classes=5)


def test_presence_vector():
    boxes = np.array([[3, .5, .5, .1, .1], [3, .2, .2, .1, .1], [0, .5, .5, .1, .1]])
    np.testing.assert_array_equal(presence_vector(boxes, 5), [1, 0, 0, 1, 0])
    assert presence_vector(np.zeros((0, 5)), 5).sum() == 0


def test_audit_counts_empties_and_leakage(tiny_dataset):
    root, names = tiny_dataset
    report, records = audit_dataset(root)
    assert report.class_names == names
    assert report.images_per_split == {"train": 3, "valid": 1, "test": 2}
    assert report.empty_images_per_split["train"] == 1
    assert report.boxes_per_class["train"]["unsealed"] == 2
    assert report.images_with_class["train"]["unsealed"] == 2
    assert report.leaked_sources == {"IMG_1": ["test", "train"]}
    assert "leakage" in report.to_text()


def test_audit_records_malformed_files(tiny_dataset):
    root, _ = tiny_dataset
    (root / "train" / "labels" / "IMG_2_jpg.rf.bbb222.txt").write_text("9 0.5 0.5 0.1 0.1\n")
    report, records = audit_dataset(root)
    assert len(report.malformed_files) == 1
    assert report.images_per_split["train"] == 2


from porkinspect.data import ImageRecord, frame_key, near_duplicate_leaks
from pathlib import Path


def _rec(name, split):
    return ImageRecord(Path(name), np.zeros((0, 5)), split)


def test_frame_key():
    assert frame_key("frame_with_red_7506") == ("frame_with_red_", 7506)
    assert frame_key("tray") == ("tray", None)


def test_near_duplicate_window():
    train = [_rec("frame_with_red_7508_jpg.rf.aa.jpg", "train"),
             _rec("frame_with_red_100_jpg.rf.bb.jpg", "train")]
    test = [_rec("frame_with_red_7506_jpg.rf.cc.jpg", "test"),   # 2 frames away
            _rec("frame_with_red_7600_jpg.rf.dd.jpg", "test"),   # 92 frames away
            _rec("other_7508_jpg.rf.ee.jpg", "test"),            # different sequence
            _rec("frame_with_red_100_jpg.rf.ff.jpg", "test")]    # exact same photo
    assert near_duplicate_leaks(train, test, window=0) == [3]
    assert near_duplicate_leaks(train, test, window=10) == [0, 3]
    assert near_duplicate_leaks(train, test, window=100) == [0, 1, 3]
    with pytest.raises(ValueError):
        near_duplicate_leaks(train, test, window=-1)


from porkinspect.data import (assign_segments, frame_gaps, segment_split,
                              unique_sources, write_yolo_split, near_duplicate_leaks)


def _seq(nums, boxes_for=lambda n: np.zeros((0, 5)), copies=1, split="train"):
    recs = []
    for n in nums:
        for c in range(copies):
            recs.append(ImageRecord(Path(f"frame_with_red_{n}_jpg.rf.{n:04x}{c}.jpg"), boxes_for(n), split))
    return recs


def test_unique_sources_groups_augmented_copies():
    recs = _seq([10, 20], copies=3)
    g = unique_sources(recs)
    assert set(g) == {"frame_with_red_10", "frame_with_red_20"} and len(g["frame_with_red_10"]) == 3


def test_frame_gaps():
    np.testing.assert_array_equal(frame_gaps(_seq([1, 3, 10, 100])), [2, 7, 90])


def test_assign_segments_cuts_at_large_gaps():
    seg = assign_segments(_seq([1, 3, 5, 50, 52, 500]), gap=10)
    ids = [seg[f"frame_with_red_{n}"] for n in (1, 3, 5, 50, 52, 500)]
    assert ids[0] == ids[1] == ids[2] and ids[3] == ids[4] and len(set(ids)) == 3
    with pytest.raises(ValueError):
        assign_segments(_seq([1]), gap=-1)


def test_segment_split_is_leak_free_and_meets_requirements():
    # 20 segments of 5 frames each, 1000 frames apart; even segments contain class 3,
    # every 4th segment is defect-free
    def boxes(n):
        s = n // 1000
        if s % 4 == 3:
            return np.zeros((0, 5))
        return np.array([[3 if s % 2 == 0 else 1, .5, .5, .1, .1]])
    nums = [s * 1000 + k * 2 for s in range(20) for k in range(5)]
    recs = _seq(nums, boxes, copies=2)
    out = segment_split(recs, gap=50, test_frac=0.2, valid_frac=0.15, must_have=[1, 3], min_pass=1)
    # no leaks within the gap, and valid/test keep one copy per photo
    assert near_duplicate_leaks(out["train"], out["test"], window=50) == []
    assert near_duplicate_leaks(out["train"], out["valid"], window=50) == []
    assert len(out["test"]) == len(unique_sources(out["test"]))
    assert all(r.split == "test" for r in out["test"])
    total = len(unique_sources(out["train"])) + len(out["valid"]) + len(out["test"])
    assert total == 100


def test_segment_split_impossible_requirements_raise():
    recs = _seq(range(0, 50, 2), lambda n: np.array([[0, .5, .5, .1, .1]]))
    with pytest.raises(RuntimeError):
        segment_split(recs, gap=5, must_have=[4], max_tries=5)


def test_write_yolo_split(tmp_path, tiny_dataset):
    root, names = tiny_dataset
    from porkinspect.data import audit_dataset
    _, rec = audit_dataset(root)
    split = {"train": rec["train"], "valid": rec["valid"], "test": rec["test"]}
    y = write_yolo_split(split, tmp_path / "out", names)
    assert y.exists() and (tmp_path / "out" / "train" / "images").is_dir()
    report, rec2 = audit_dataset(tmp_path / "out")
    assert report.images_per_split == {"train": 3, "valid": 1, "test": 2}
    assert report.boxes_per_class["train"]["unsealed"] == 2
