"""Baseline model: a small CNN trained from scratch for tray-level classification.

The architecture follows the convolution-block design used in my ECE4179 Assignment 1
(Q4, GTSRB traffic-sign classifier): repeated ``Conv2d -> BatchNorm2d -> ReLU`` blocks
with pooling, adapted here from single-label classification (softmax, 43 classes) to
**multi-label** classification (sigmoid, one output per defect type), because one tray
can contain several defect types at once.

It predicts *whether* each defect type is present, but not *where*. It is used as the
baseline that the YOLO detector (`porkinspect.detector`) must beat to justify its extra
complexity.
"""

from __future__ import annotations

import copy
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .data import ImageRecord, presence_vector
from .io_utils import load_image

#: Network input size (height, width). Keeps the 4:3 aspect ratio of 720 x 540 images.
INPUT_HW: Tuple[int, int] = (240, 320)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def conv_block(c_in: int, c_out: int) -> nn.Sequential:
    """Conv(3x3) -> BatchNorm -> ReLU -> MaxPool(2).

    Parameters
    ----------
    c_in, c_out : int
        Input and output channels.

    Returns
    -------
    torch.nn.Sequential
    """
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class TrayCNN(nn.Module):
    """Five-block CNN with global average pooling and a multi-label head.

    Parameters
    ----------
    n_classes : int
        Number of defect types (outputs).
    widths : sequence of int
        Output channels of each convolution block.
    dropout : float
        Dropout probability before the final linear layer.

    Notes
    -----
    ``forward`` returns raw logits of shape (batch, n_classes); apply
    ``torch.sigmoid`` to get presence probabilities.
    """

    def __init__(self, n_classes: int = 5, widths: Sequence[int] = (32, 64, 128, 128, 256),
                 dropout: float = 0.3) -> None:
        super().__init__()
        blocks, c = [], 3
        for w in widths:
            blocks.append(conv_block(c, w))
            c = w
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(c, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self.head(self.pool(self.features(x)))


def preprocess(image: np.ndarray, hw: Tuple[int, int] = INPUT_HW) -> torch.Tensor:
    """Convert a validated RGB uint8 image to a normalised (3, H, W) tensor.

    Parameters
    ----------
    image : numpy.ndarray
        Shape (H, W, 3), dtype uint8.
    hw : tuple of int
        Output (height, width).

    Returns
    -------
    torch.Tensor
        Float tensor of shape (3, hw[0], hw[1]), ImageNet-normalised.
    """
    from PIL import Image

    im = Image.fromarray(image).resize((hw[1], hw[0]), Image.BILINEAR)
    x = (np.asarray(im, dtype=np.float32) / 255.0 - _MEAN) / _STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


class TrayDataset(Dataset):
    """PyTorch dataset of (image tensor, presence vector) pairs.

    Parameters
    ----------
    records : sequence of ImageRecord
        Images and boxes (from `porkinspect.data.scan_split`).
    n_classes : int
        Number of defect classes.
    augment : bool
        If True, apply random horizontal/vertical flips and brightness/contrast jitter
        (training only). Flips do not change which defects are present.
    hw : tuple of int
        Network input size.
    """

    def __init__(self, records: Sequence[ImageRecord], n_classes: int,
                 augment: bool = False, hw: Tuple[int, int] = INPUT_HW) -> None:
        self.records = list(records)
        self.n_classes = n_classes
        self.augment = augment
        self.hw = hw

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        r = self.records[idx]
        image, _ = load_image(r.image_path)
        x = preprocess(image, self.hw)
        if self.augment:
            if torch.rand(1) < 0.5:
                x = torch.flip(x, dims=[2])
            if torch.rand(1) < 0.5:
                x = torch.flip(x, dims=[1])
            x = x * (0.8 + 0.4 * torch.rand(1)) + 0.2 * (torch.rand(1) - 0.5)
        y = torch.from_numpy(presence_vector(r.boxes, self.n_classes))
        return x, y


def positive_weights(records: Sequence[ImageRecord], n_classes: int) -> torch.Tensor:
    """Per-class ``pos_weight`` for BCEWithLogitsLoss to counter class imbalance.

    weight_c = (#trays without class c) / (#trays with class c), clipped to [1, 20].

    Parameters
    ----------
    records : sequence of ImageRecord
        Training records.
    n_classes : int
        Number of classes.

    Returns
    -------
    torch.Tensor
        Shape (n_classes,).
    """
    y = np.stack([presence_vector(r.boxes, n_classes) for r in records])
    pos = y.sum(axis=0)
    neg = len(y) - pos
    w = np.where(pos > 0, neg / np.maximum(pos, 1), 1.0)
    return torch.tensor(np.clip(w, 1.0, 20.0), dtype=torch.float32)


@torch.no_grad()
def predict_scores(model: nn.Module, loader: DataLoader,
                   device: str = "cpu") -> Tuple[np.ndarray, np.ndarray]:
    """Run the model over a loader.

    Parameters
    ----------
    model : torch.nn.Module
        Trained `TrayCNN`.
    loader : torch.utils.data.DataLoader
        Yields (x, y) batches.
    device : str
        ``"cpu"`` or ``"cuda"``.

    Returns
    -------
    scores : numpy.ndarray
        Shape (n, n_classes), sigmoid probabilities.
    targets : numpy.ndarray
        Shape (n, n_classes), ground-truth presence.
    """
    model.eval().to(device)
    s, t = [], []
    for x, y in loader:
        s.append(torch.sigmoid(model(x.to(device))).cpu().numpy())
        t.append(y.numpy())
    return np.concatenate(s), np.concatenate(t)


def train_baseline(train_records: Sequence[ImageRecord], val_records: Sequence[ImageRecord],
                   n_classes: int, epochs: int = 25, batch_size: int = 32, lr: float = 1e-3,
                   weight_decay: float = 1e-4, device: Optional[str] = None,
                   num_workers: int = 2, seed: int = 0, verbose: bool = True
                   ) -> Tuple[TrayCNN, Dict[str, List[float]]]:
    """Train `TrayCNN` and keep the checkpoint with the best validation macro-F1.

    Parameters
    ----------
    train_records, val_records : sequence of ImageRecord
        Training and validation data. The test split is never used here.
    n_classes : int
        Number of defect classes.
    epochs, batch_size : int
        Training length and batch size.
    lr, weight_decay : float
        AdamW learning rate and weight decay (L2 regularisation).
    device : str, optional
        Defaults to ``"cuda"`` if available.
    num_workers : int
        DataLoader worker processes.
    seed : int
        Random seed for repeatability.
    verbose : bool
        Print one line per epoch.

    Returns
    -------
    model : TrayCNN
        Best model (by validation macro-F1 at threshold 0.5).
    history : dict of list
        ``train_loss``, ``val_loss``, ``val_macro_f1``, ``epoch_time_s`` per epoch.
    """
    from .metrics import multilabel_report

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tr = DataLoader(TrayDataset(train_records, n_classes, augment=True),
                    batch_size=batch_size, shuffle=True, num_workers=num_workers)
    va = DataLoader(TrayDataset(val_records, n_classes), batch_size=batch_size,
                    num_workers=num_workers)
    model = TrayCNN(n_classes).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=positive_weights(train_records, n_classes).to(device))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    hist: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "val_macro_f1": [],
                                    "epoch_time_s": []}
    best, best_f1 = None, -1.0
    names = [str(i) for i in range(n_classes)]
    for ep in range(epochs):
        t0 = time.perf_counter()
        model.train()
        tot, n = 0.0, 0
        for x, y in tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            tot += loss.item() * len(x)
            n += len(x)
        sched.step()
        model.eval()
        vl, vn = 0.0, 0
        with torch.no_grad():
            for x, y in va:
                x, y = x.to(device), y.to(device)
                vl += loss_fn(model(x), y).item() * len(x)
                vn += len(x)
        scores, targets = predict_scores(model, va, device)
        f1 = multilabel_report(targets, scores >= 0.5, names)["macro_f1"]
        hist["train_loss"].append(tot / max(n, 1))
        hist["val_loss"].append(vl / max(vn, 1))
        hist["val_macro_f1"].append(f1)
        hist["epoch_time_s"].append(time.perf_counter() - t0)
        if f1 > best_f1:
            best_f1, best = f1, copy.deepcopy(model.state_dict())
        if verbose:
            print(f"epoch {ep + 1:3d}/{epochs}  train {hist['train_loss'][-1]:.4f}  "
                  f"val {hist['val_loss'][-1]:.4f}  val macro-F1 {f1:.3f}  "
                  f"({hist['epoch_time_s'][-1]:.1f}s)")
    if best is not None:
        model.load_state_dict(best)
    return model, hist
