"""Generates run_pipeline.ipynb (kept as a script so the notebook is reproducible)."""
import json
from pathlib import Path

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": s.strip("\n")})


md(r"""
# Pork rasher tray inspection: full pipeline (MMA3001 project)

Runs the whole project end to end on Google Colab:

1. Set up: clone the repo, install packages
2. Download the dataset (Roboflow, YOLOv8 format)
3. **Audit** the data: class balance, defect-free trays, leakage
4. Train the **main model**: YOLOv8n detector (pretrained, fine-tuned)
5. Train the **baseline**: from-scratch CNN tray classifier (ECE4179 Q4 style)
6. **Validate**: choose the threshold on *validation*, report on *test*
7. **Sensitivity analysis**: decision threshold and input resolution
8. **Performance**: latency, FLOPs, arithmetic intensity, line-speed budget
9. **Error investigation**: occlusion sensitivity maps
10. Save everything in `results/` and download it

**Before you start:** Runtime → Change runtime type → **T4 GPU**. Then Runtime → Run all.
Total time is about 60–90 min, mostly YOLO training.
""")

code(r"""
# 1. SETUP: works on Colab, Kaggle, or locally (VS Code, opened from the repo's notebooks/ folder)
GITHUB_USER = "anhtruong27"
REPO = "pork-rasher-inspection"

import os, sys, shutil
from pathlib import Path
here = Path.cwd()
LOCAL = True
if (here / "src" / "porkinspect").exists():                 # already at the repo root (local)
    pass
elif (here.parent / "src" / "porkinspect").exists():        # opened from notebooks/ (local)
    os.chdir(here.parent)
else:                                                        # Colab / Kaggle: get the code from GitHub
    LOCAL = False
    if not os.path.exists(REPO):
        !git clone https://github.com/{GITHUB_USER}/{REPO}.git
    os.chdir(REPO)
%pip install -q ultralytics roboflow
sys.path.insert(0, "src")

# Settings
OUT = "results/run1_local" if LOCAL else "results"   # locally: never overwrite the committed run-1 results
DATA_DIR = "data"     # local Windows: use a short path with only English letters, e.g. "C:/mma_data"
BATCH = 16            # lower to 8 if you get "CUDA out of memory" (6 GB laptop GPU)
WORKERS = 2           # image-loading processes; Ultralytics' default (8) can run out of RAM on a laptop
os.makedirs(OUT, exist_ok=True)

import torch
print("Working in", os.getcwd(), "| results ->", OUT)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE - switch runtime to T4 GPU (Colab) / select a CUDA kernel (local)!")
""")

code(r"""
# 2. DOWNLOAD DATASET. Paste your Roboflow private API key when asked (it is NOT saved).
from getpass import getpass
from roboflow import Roboflow
from pathlib import Path
import yaml

rf = Roboflow(api_key=getpass("Roboflow API key: "))
project = rf.workspace("hello-2aqe0").project("pork-rasher-error-packaging")
DATASET_VERSION = 4
ds = project.version(DATASET_VERSION).download("yolov8", location=f"{DATA_DIR}/pork", overwrite=True)

# Rewrite data.yaml with absolute paths so Ultralytics finds every split.
DATA = Path(f"{DATA_DIR}/pork").resolve()
cfg = yaml.safe_load((DATA / "data.yaml").read_text())
cfg.update({"path": str(DATA), "train": "train/images", "val": "valid/images", "test": "test/images"})
(DATA / "data.yaml").write_text(yaml.safe_dump(cfg))
CLASS_NAMES = list(cfg["names"]) if isinstance(cfg["names"], list) else [cfg["names"][k] for k in sorted(cfg["names"])]
N = len(CLASS_NAMES)
print(CLASS_NAMES)
""")

code(r"""
# 3. DATA AUDIT. Answers: class imbalance? defect-free trays? leakage between splits?
from porkinspect.data import audit_dataset
report, records = audit_dataset(DATA, CLASS_NAMES)
txt = report.to_text()
print(txt)
Path(f"{OUT}/dataset_audit.txt").write_text(txt)
""")

md(r"""
**Read the audit before continuing:**
- *No defect-free trays* → the tray PASS/FAIL accuracy is not meaningful. Report the per-class "which defects are present" metrics instead, and say so in the report as a limitation.
- *Leakage > 0* → augmented copies of the same photo are in more than one split, so test scores are optimistic. Report it honestly. The optional cell below builds a leak-free test subset.
""")

code(r"""
# 3b. Leak-free test subset: test images whose original photo never appears in train/valid.
train_val_sources = {r.source_id for s in ("train", "valid") for r in records.get(s, [])}
test_clean = {str(r.image_path) for r in records["test"] if r.source_id not in train_val_sources}
print(f"test images: {len(records['test'])}, leak-free: {len(test_clean)}")
""")

code(r"""
# 4. MAIN MODEL: fine-tune YOLOv8n (pretrained on COCO). Best epoch chosen on VALIDATION mAP.
import time, json
from porkinspect.detector import train_detector, box_metrics, predict_detections, detections_to_scores, model_gflops

def newest(pattern):   # most recent file/folder under runs/ matching pattern (Ultralytics may nest folders)
    hits = sorted(Path("runs").resolve().rglob(pattern), key=lambda p: p.stat().st_mtime)
    if not hits:
        raise FileNotFoundError(f"nothing matching {pattern} under runs/")
    return hits[-1]
def run_dir(run_name):   # training folder of a run, found by name wherever Ultralytics put it
    return newest(f"{run_name}/weights").parent

EPOCHS, IMGSZ = 40, 640
t0 = time.perf_counter()
yolo = train_detector(DATA / "data.yaml", "yolov8n.pt", epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, workers=WORKERS)
yolo_train_min = (time.perf_counter() - t0) / 60
print(f"YOLO training time: {yolo_train_min:.1f} min")
shutil.copy(run_dir("yolo") / "results.png", f"{OUT}/yolo_training_curves.png")
shutil.copy(run_dir("yolo") / "weights" / "best.pt", f"{OUT}/yolo_best.pt")
""")

code(r"""
# Box-level metrics (mAP) on validation and test
yolo_box = {"val": box_metrics(yolo, DATA / "data.yaml", "val", IMGSZ),
            "test": box_metrics(yolo, DATA / "data.yaml", "test", IMGSZ)}
print(json.dumps(yolo_box, indent=2))
shutil.copy(newest("confusion_matrix_normalized.png"), f"{OUT}/yolo_confusion_matrix_test.png")   # the test run was last
""")

code(r"""
# Tray-level presence scores from the detector (max box confidence per class)
import numpy as np
from porkinspect.data import presence_vector

def truth(recs): return np.stack([presence_vector(r.boxes, N) for r in recs])
def yolo_scores(recs): return detections_to_scores(predict_detections(yolo, [r.image_path for r in recs], imgsz=IMGSZ), N)

Y = {s: truth(records[s]) for s in ("valid", "test")}
S_yolo = {s: yolo_scores(records[s]) for s in ("valid", "test")}
""")

code(r"""
# 5. BASELINE: from-scratch CNN tray classifier (no localisation)
from porkinspect.baseline import train_baseline, predict_scores, TrayDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

t0 = time.perf_counter()
cnn, hist = train_baseline(records["train"], records["valid"], N, epochs=25, batch_size=32, num_workers=WORKERS)
cnn_train_min = (time.perf_counter() - t0) / 60
torch.save(cnn.state_dict(), f"{OUT}/cnn_best.pt")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(hist["train_loss"], label="train"); ax[0].plot(hist["val_loss"], label="validation")
ax[0].set(xlabel="epoch", ylabel="BCE loss", title="Baseline CNN loss"); ax[0].legend()
ax[1].plot(hist["val_macro_f1"]); ax[1].set(xlabel="epoch", ylabel="macro F1", title="Baseline CNN validation macro F1")
plt.tight_layout(); plt.savefig(f"{OUT}/cnn_training_curves.png", dpi=150); plt.show()

S_cnn = {s: predict_scores(cnn, DataLoader(TrayDataset(records[s], N), batch_size=64), "cuda")[0] for s in ("valid", "test")}
""")

code(r"""
# 6. VALIDATION: choose each model's threshold on VALIDATION, then evaluate once on TEST.
from porkinspect.metrics import threshold_sweep, best_threshold, multilabel_report
TH = np.round(np.arange(0.05, 0.96, 0.05), 2)
results = {}
for name, S in [("YOLOv8n detector", S_yolo), ("Baseline CNN", S_cnn)]:
    sw_val = threshold_sweep(Y["valid"], S["valid"], TH)
    thr = best_threshold(sw_val, "macro_f1")
    rep = multilabel_report(Y["test"], S["test"] >= thr, CLASS_NAMES)
    results[name] = {"threshold": thr, "test": rep, "sweep_test": {k: v.tolist() for k, v in threshold_sweep(Y["test"], S["test"], TH).items()}}
    print(f"\n=== {name}  (threshold {thr} chosen on validation) ===")
    print(f"macro F1 {rep['macro_f1']:.3f} | macro recall (balanced) {rep['macro_recall']:.3f} | micro F1 {rep['micro']['f1']:.3f} | exact match {rep['exact_match']:.3f}")
    print(f"tray: accuracy {rep['tray']['accuracy']:.3f}, defective trays caught {rep['tray']['fail_recall']:.3f}, good trays in test {rep['tray']['n_pass_trays']}")
    for c, m in rep["per_class"].items():
        print(f"   {c:16s} P {m['precision']:.2f}  R {m['recall']:.2f}  F1 {m['f1']:.2f}  (n={m['support']})")

# Leak-free subset check for the detector
if len(test_clean):
    idx = [i for i, r in enumerate(records["test"]) if str(r.image_path) in test_clean]
    rc = multilabel_report(Y["test"][idx], S_yolo["test"][idx] >= results["YOLOv8n detector"]["threshold"], CLASS_NAMES)
    results["YOLOv8n detector"]["test_leak_free_macro_f1"] = rc["macro_f1"]
    print("\nYOLO macro F1 on leak-free test subset:", round(rc["macro_f1"], 3))
""")

code(r"""
# 7a. SENSITIVITY: decision threshold (test split, for illustration only; the threshold was chosen on validation)
fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for a, name in zip(ax, results):
    sw = results[name]["sweep_test"]
    a.plot(sw["threshold"], sw["micro_precision"], label="precision")
    a.plot(sw["threshold"], sw["micro_recall"], label="recall")
    a.plot(sw["threshold"], sw["macro_f1"], label="macro F1")
    a.axvline(results[name]["threshold"], ls="--", c="grey", label="chosen (val)")
    a.set(title=name, xlabel="decision threshold"); a.legend()
ax[0].set_ylabel("score")
plt.tight_layout(); plt.savefig(f"{OUT}/threshold_sensitivity.png", dpi=150); plt.show()
""")

code(r"""
# 7b. SENSITIVITY / OPTIMISATION: input resolution vs accuracy vs speed (detector)
from porkinspect.performance import time_function
from porkinspect.io_utils import load_image
sample = load_image(records["test"][0].image_path)[0][:, :, ::-1].copy()
res_rows = []
for sz in (320, 480, 640):
    m = box_metrics(yolo, DATA / "data.yaml", "test", sz)
    t = time_function(lambda: yolo.predict(sample, imgsz=sz, verbose=False), warmup=5, repeats=30)
    res_rows.append({"imgsz": sz, "map50": m["map50"], "map50_95": m["map50_95"], "gpu_ms": t["median_ms"]})
    print(res_rows[-1])
results["resolution_study"] = res_rows
""")

code(r"""
# 8. PERFORMANCE: latency (GPU and CPU, batch 1), FLOPs, arithmetic intensity, line budget
from porkinspect.performance import count_flops, required_throughput
from porkinspect.baseline import preprocess
x = preprocess(load_image(records["test"][0].image_path)[0]).unsqueeze(0)

perf = {}
cnn.eval().cuda()
perf["Baseline CNN"] = {
    "gpu_ms": time_function(lambda: cnn(x.cuda()), 10, 50)["median_ms"],
    **count_flops(cnn, tuple(x.shape)),
    "train_min": cnn_train_min,
}
perf["YOLOv8n detector"] = {
    "gpu_ms": time_function(lambda: yolo.predict(sample, imgsz=IMGSZ, verbose=False), 10, 50)["median_ms"],
    "gflops_reported": model_gflops(yolo),
    "params": float(sum(p.numel() for p in yolo.model.parameters())),
    "train_min": yolo_train_min,
}
# CPU timing (a factory PC may have no GPU)
cnn_cpu = cnn.cpu()
perf["Baseline CNN"]["cpu_ms"] = time_function(lambda: cnn_cpu(x), 3, 20, sync_cuda=False)["median_ms"]
perf["YOLOv8n detector"]["cpu_ms"] = time_function(lambda: yolo.predict(sample, imgsz=IMGSZ, device="cpu", verbose=False), 3, 20, sync_cuda=False)["median_ms"]
cnn.cuda()

LINE_SPEED = 60  # trays per minute: an assumed line speed, state it in the report
perf["line_budget"] = required_throughput(LINE_SPEED)
results["performance"] = perf
print(json.dumps(perf, indent=2))
""")

code(r"""
# 9. ERROR INVESTIGATION: occlusion sensitivity on 3 correct + 3 wrong test trays (detector)
from porkinspect.occlusion import occlusion_map, overlay
thr = results["YOLOv8n detector"]["threshold"]
pred = S_yolo["test"] >= thr
correct = [i for i in range(len(pred)) if (pred[i] == Y["test"][i].astype(bool)).all() and Y["test"][i].any()]
wrong = [i for i in range(len(pred)) if not (pred[i] == Y["test"][i].astype(bool)).all()]
chosen = [(i, True) for i in correct[:3]] + [(i, False) for i in wrong[:3]]

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for ax, (i, ok) in zip(axes.ravel(), chosen):
    img = load_image(records["test"][i].image_path)[0]
    # class of interest: the true class with the largest score error (or the top true class)
    true_c = np.flatnonzero(Y["test"][i]); c = int(true_c[np.argmax(S_yolo["test"][i][true_c])]) if len(true_c) else int(np.argmax(S_yolo["test"][i]))
    def score(im, c=c):
        r = yolo.predict(im[:, :, ::-1].copy(), conf=0.01, imgsz=IMGSZ, verbose=False)[0]
        cf = r.boxes.conf.cpu().numpy()[r.boxes.cls.cpu().numpy() == c]
        return float(cf.max()) if len(cf) else 0.0
    heat, base = occlusion_map(score, img, patch=90, stride=45)
    overlay(img, heat, ax=ax, title=f"{'CORRECT' if ok else 'WRONG'} | {CLASS_NAMES[c]} | score {base:.2f}")
plt.tight_layout(); plt.savefig(f"{OUT}/occlusion_maps.png", dpi=130); plt.show()
""")

code(r"""
# 9b. ROBUSTNESS OF THE VALIDATION: near-duplicate video frames + bootstrap confidence intervals
if not LOCAL:
    !git pull -q
import importlib, porkinspect.data, porkinspect.metrics
importlib.reload(porkinspect.data); importlib.reload(porkinspect.metrics)
from porkinspect.data import near_duplicate_leaks
from porkinspect.metrics import bootstrap_ci, multilabel_report

MODELS = {"YOLOv8n detector": S_yolo, "Baseline CNN": S_cnn}
GOOD = [c for c in CLASS_NAMES if c in ("packaging-error", "unsealed")]   # classes with enough test cases
gi = [CLASS_NAMES.index(c) for c in GOOD]
robust = {"leakage_windows": [], "bootstrap": {}}

# (a) How many test trays have a near-identical training frame, and do scores drop without them?
print("window | leaked test trays | leak-free n | " + " | ".join(f"{m} macroF1 / tray acc" for m in MODELS))
for w in (0, 2, 10, 30, 100):
    leaked = set(near_duplicate_leaks(records["train"] + records["valid"], records["test"], window=w))
    keep = [i for i in range(len(records["test"])) if i not in leaked]
    row = {"window": w, "leaked": len(leaked), "n_clean": len(keep)}
    cells = []
    for m, S in MODELS.items():
        r = multilabel_report(Y["test"][keep], S["test"][keep] >= results[m]["threshold"], CLASS_NAMES)
        row[m] = {"macro_f1": r["macro_f1"], "tray_accuracy": r["tray"]["accuracy"]}
        cells.append(f"{r['macro_f1']:.3f} / {r['tray']['accuracy']:.3f}")
    robust["leakage_windows"].append(row)
    print(f"{w:6d} | {len(leaked):17d} | {len(keep):11d} | " + " | ".join(cells))

# (b) 95 % bootstrap confidence intervals on the full test set (80 trays)
for m, S in MODELS.items():
    P = S["test"] >= results[m]["threshold"]
    robust["bootstrap"][m] = {
        "macro_f1_all": bootstrap_ci(Y["test"], P, lambda t, p: multilabel_report(t, p, CLASS_NAMES)["macro_f1"]),
        "macro_f1_main_classes": bootstrap_ci(Y["test"][:, gi], P[:, gi], lambda t, p: multilabel_report(t, p, GOOD)["macro_f1"]),
        "tray_accuracy": bootstrap_ci(Y["test"], P, lambda t, p: multilabel_report(t, p, CLASS_NAMES)["tray"]["accuracy"]),
        "defective_trays_caught": bootstrap_ci(Y["test"], P, lambda t, p: multilabel_report(t, p, CLASS_NAMES)["tray"]["fail_recall"]),
    }
    print(f"\n{m} (95% CI, 2000 resamples)")
    for k, v in robust["bootstrap"][m].items():
        print(f"   {k:24s} {v['estimate']:.3f}  [{v['low']:.3f}, {v['high']:.3f}]")

Path(f"{OUT}/robustness.json").write_text(json.dumps(robust, indent=2, default=float))
np.savez(f"{OUT}/scores.npz", Y_valid=Y["valid"], Y_test=Y["test"],
         yolo_valid=S_yolo["valid"], yolo_test=S_yolo["test"], cnn_valid=S_cnn["valid"], cnn_test=S_cnn["test"],
         test_files=np.array([r.image_path.name for r in records["test"]]))
print(f"\nsaved {OUT}/robustness.json and {OUT}/scores.npz")
""")

code(r"""
# 10. SAVE: summary table + JSON, then download results.zip
rows = ["| Model | Threshold (val) | Test macro F1 | Test macro recall | Test micro F1 | Exact match | GPU ms/img | CPU ms/img |",
        "|---|---|---|---|---|---|---|---|"]
for name in ("YOLOv8n detector", "Baseline CNN"):
    r, p = results[name], results["performance"][name]
    t = r["test"]
    rows.append(f"| {name} | {r['threshold']} | {t['macro_f1']:.3f} | {t['macro_recall']:.3f} | {t['micro']['f1']:.3f} | {t['exact_match']:.3f} | {p['gpu_ms']:.1f} | {p['cpu_ms']:.1f} |")
table = "\n".join(rows)
print(table)
Path(f"{OUT}/comparison_table.md").write_text(table + "\n")
results["yolo_box_metrics"] = yolo_box
Path(f"{OUT}/results.json").write_text(json.dumps(results, indent=2, default=float))

if LOCAL:
    print("Results are already in", os.path.abspath(OUT))
else:
    shutil.make_archive("results", "zip", root_dir=".", base_dir=OUT)
    try:                                   # Google Colab: download straight away
        from google.colab import files
        files.download("results.zip")
    except ImportError:                    # Kaggle: use the Output panel
        print("Download results.zip from the Output panel:", os.path.abspath("results.zip"))
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
      "kernelspec": {"name": "python3", "display_name": "Python 3"}},
      "nbformat": 4, "nbformat_minor": 0}
out = Path(__file__).with_name("run_pipeline.ipynb")
out.write_text(json.dumps(nb, indent=1))
print("wrote", out)
