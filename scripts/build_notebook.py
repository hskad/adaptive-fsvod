"""Build the Colab/Kaggle notebook with ordinary Python cells."""

import hashlib
import json
from pathlib import Path


cells = []


def cell(kind, source):
    source = source.strip() + "\n"
    entry = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True),
             "id": hashlib.sha1(source.encode()).hexdigest()[:12]}
    if kind == "code":
        entry.update({"execution_count": None, "outputs": []})
    cells.append(entry)


cell("markdown", """
# Adaptive FSVOD: GPU pilot

Run one cell at a time. Select a GPU runtime in Colab or Kaggle. Internet access is needed for package and model downloads. This notebook uses a frozen public OWLv2 Base encoder and trains small temporal heads.

The repository contains a reduced adaptation of the assigned paper. It has no measured real-video results yet. Synthetic software-check results are not research results.

**Data:** the notebook downloads FSYTV annotations and only the selected RGB frames from a pinned public YouTube-VOS mirror. It uses HTTP byte ranges and refuses a full archive download. Frame paths, ZIP CRCs, dimensions, and per-image hashes are checked. See `docs/DATA.md` for sources and terms.

**Persistence:** set `EXPERIMENT_DIR` to mounted Google Drive or another persistent directory before extracting features. Free runtimes can disconnect. Save the run before closing a session.
""")

cell("code", """
import os
import subprocess
import sys
from pathlib import Path

os.environ["USE_TF"] = "0"
BASE = Path("/kaggle/working") if Path("/kaggle").exists() else Path("/content")
REPO = BASE / "adaptive-fsvod"
if not (REPO / ".git").exists():
    subprocess.run(["git", "clone", "https://github.com/hskad/adaptive-fsvod.git", str(REPO)], check=True)
os.chdir(REPO)
subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
# Editable installation affects new Python processes. Also expose src to the
# notebook kernel, which was already running when pip created its .pth file.
SOURCE_PATH = str(REPO / "src")
if SOURCE_PATH not in sys.path:
    sys.path.insert(0, SOURCE_PATH)
import importlib
importlib.invalidate_caches()
import adaptive_fsvod
print("Project imports:", adaptive_fsvod.__file__)
subprocess.run(["git", "rev-parse", "HEAD"], check=True)
""")

cell("code", """
import torch
assert torch.cuda.is_available(), "Select a GPU runtime before feature extraction."
print("GPU:", torch.cuda.get_device_name(0))
print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda)
subprocess.run([sys.executable, "-m", "pytest", "-q"], check=True)
""")

cell("markdown", """
## Set paths

For Colab persistence, mount Drive in a separate cell with `from google.colab import drive; drive.mount('/content/drive')`. Then set `EXPERIMENT_DIR`, for example, to `/content/drive/MyDrive/adaptive-fsvod-experiment`. Do not put experiment outputs in a read-only Kaggle input directory.
""")

cell("code", """
EXPERIMENT_DIR = BASE / "adaptive-fsvod-experiment"  # Change to persistent storage.
IMAGE_ROOT = EXPERIMENT_DIR / "images"             # Selected frames download here.
SEED = 42
STEPS = 300
EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

def run(module, **flags):
    command = [sys.executable, "-m", "adaptive_fsvod." + module]
    for name, value in flags.items():
        if isinstance(value, bool):
            if value:
                command.append("--" + name.replace("_", "-"))
            continue
        command.append("--" + name.replace("_", "-"))
        if isinstance(value, (list, tuple)):
            command.extend(map(str, value))
        else:
            command.append(str(value))
    subprocess.run(command, check=True)

""")

cell("code", """
RAW = EXPERIMENT_DIR / "raw"
RAW.mkdir(exist_ok=True)
ARCHIVE = RAW / "fsytv-annotations.zip"
BOXES = RAW / "fsytv-boxes.json"
if not ARCHIVE.exists():
    subprocess.run([sys.executable, "-m", "gdown", "1-m5Qcuxj4h8rryqnELtgnvOtTtWJ_Vys", "-O", str(ARCHIVE)], check=True)
if not BOXES.exists():
    run("convert_fsytv", archive=ARCHIVE, output=BOXES)

DATA = EXPERIMENT_DIR / "pilot"
run("prepare", annotation=BOXES, images_root=IMAGE_ROOT, output=DATA,
    class_counts=[6, 3, 3], max_videos_per_class=3, frame_stride=4, seed=SEED, skip_image_check=True)
run("download_frames", manifests=[DATA / f"{split}.json" for split in ["train", "val", "test"]],
    images_root=IMAGE_ROOT, range_cache=EXPERIMENT_DIR / "download_ranges")
from adaptive_fsvod.data import VideoManifest
for split in ["train", "val", "test"]:
    print(split, VideoManifest(DATA / f"{split}.json").validate_images())
""")

cell("markdown", """
## Cache frozen features

All methods share these caches. The encoder runs at its native resolution. Only the top 128 tokens are retained afterward. The cache can resume an interrupted extraction. Settings and annotation checksums must match.
""")

cell("code", """
from adaptive_fsvod.cache import FrozenEncoder, build_cache
from adaptive_fsvod.data import VideoManifest

CACHE = EXPERIMENT_DIR / "cache"
encoder = FrozenEncoder(device="cuda", revision="cfd3195ba4ea9592eec887ded089f4c08eff231d", candidates=128)
for split in ["train", "val", "test"]:
    print(split, build_cache(VideoManifest(DATA / f"{split}.json"), CACHE / split, encoder, per_class=16))
del encoder
torch.cuda.empty_cache()
""")

cell("code", """
for split, count, episode_seed in [("train", 300, 42), ("val", 20, 43), ("test", 30, 44)]:
    flags = dict(manifest=DATA / f"{split}.json", cache=CACHE / split,
                 output=DATA / f"{split}_episodes.json", episodes=count,
                 n_way=2, k_shot=1, seed=episode_seed)
    if split == "train":
        flags["max_frames"] = 4
    run("episodes", **flags)
""")

cell("markdown", """
## Train

First test a short overfit run in a separate output directory. Inspect detections and loss trends. A finite loss alone is insufficient. Train both methods with equal budgets. The initial settings below have not been selected on validation data. Use validation experiments to tune both thresholds with equal search budgets. Do not use test AP for model selection.

To resume a disconnected run, add `resume=<run-directory>/last.pt` to its flags and retain the original total budget. Avoid a new run in an existing directory.
""")

cell("code", """
RUNS = EXPERIMENT_DIR / "runs"
for method in ["fixed", "adaptive"]:
    run("train", train_manifest=DATA / "train.json", val_manifest=DATA / "val.json",
        train_cache=CACHE / "train", val_cache=CACHE / "val",
        train_episodes=DATA / "train_episodes.json", val_episodes=DATA / "val_episodes.json",
        config=REPO / f"configs/{method}.json", output=RUNS / f"{method}_seed{SEED}",
        steps=STEPS, eval_every=50, seed=SEED, device="cuda")
""")

cell("markdown", """
## Final evaluation

Run after validation choices are fixed. This is frame-level box AP, not video mask AP. Cached pipeline FPS excludes the encoder. Use the inference command in `docs/RUNBOOK.md` for an end-to-end timing sample. With time available, repeat the paired training with additional seeds and summarize variability.
""")

cell("code", """
for method in ["fixed", "adaptive"]:
    directory = RUNS / f"{method}_seed{SEED}"
    run("evaluate", manifest=DATA / "test.json", cache=CACHE / "test",
        episodes=DATA / "test_episodes.json", checkpoint=directory / "best.pt",
        output=directory / "test", device="cuda")
run("compare", fixed=RUNS / f"fixed_seed{SEED}/test/metrics.json",
    adaptive=RUNS / f"adaptive_seed{SEED}/test/metrics.json", output=RUNS / f"comparison_seed{SEED}")
""")

cell("code", """
from IPython.display import display
from PIL import Image
display(Image.open(RUNS / f"comparison_seed{SEED}/comparison.png"))
print((RUNS / f"comparison_seed{SEED}/comparison.csv").read_text())
""")

cell("markdown", """
## Preserve evidence

Save `best.pt`, `last.pt`, configurations, training logs, evaluation JSON, selected data manifests, fixed episode files, and the Git commit. The configs record actual package and hardware versions. Publish accessible checkpoint links only after verifying the files. Do not upload source dataset images to GitHub. Report measured negative results as well as positive results.
""")

notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"}, "colab": {"provenance": []}}, "nbformat": 4, "nbformat_minor": 5}
destination = Path(__file__).resolve().parents[1] / "notebooks/colab_kaggle_pilot.ipynb"
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
print(destination)
