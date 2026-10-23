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
