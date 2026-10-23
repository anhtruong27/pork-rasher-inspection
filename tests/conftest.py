"""Shared fixtures: a tiny synthetic dataset in Roboflow YOLOv8 layout."""
import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def tiny_dataset(tmp_path):
    """Create train/valid/test splits with known boxes, one empty image and one leak."""
    root = tmp_path / "ds"
    names = ["loose-meat", "packaging-error", "twisted-meat", "unsealed", "wrinkle"]
    (root).mkdir()
    (root / "data.yaml").write_text(
        "nc: 5\nnames: ['loose-meat', 'packaging-error', 'twisted-meat', 'unsealed', 'wrinkle']\n")
    files = {
        "train": {"IMG_1_jpg.rf.aaa111": "0 0.5 0.5 0.2 0.2\n3 0.3 0.3 0.1 0.1\n",
                  "IMG_2_jpg.rf.bbb222": "3 0.5 0.5 0.4 0.4\n",
                  "IMG_3_jpg.rf.ccc333": ""},                       # defect-free tray
        "valid": {"IMG_4_jpg.rf.ddd444": "4 0.5 0.5 0.2 0.2\n"},
        "test": {"IMG_5_jpg.rf.eee555": "1 0.5 0.5 0.2 0.2\n",
                 "IMG_1_jpg.rf.fff666": "0 0.5 0.5 0.2 0.2\n"},      # leak of IMG_1
    }
    for split, items in files.items():
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
        for stem, lbl in items.items():
            Image.fromarray(np.full((540, 720, 3), 100, np.uint8)).save(
                root / split / "images" / f"{stem}.jpg")
            (root / split / "labels" / f"{stem}.txt").write_text(lbl)
    return root, names
