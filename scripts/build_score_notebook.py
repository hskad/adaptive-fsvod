"""Build the validation-first score-calibration reproduction notebook."""

import copy
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
source = json.loads((root / "notebooks/colab_kaggle_pilot.ipynb").read_text(encoding="utf-8"))
last = next(i for i, c in enumerate(source["cells"])
            if c["cell_type"] == "code" and "for split, count, episode_seed" in "".join(c["source"]))
notebook = copy.deepcopy(source)
notebook["cells"] = notebook["cells"][:last + 1]


def cell(kind, text):
    text = text.strip() + "\n"
    result = {"cell_type": kind, "metadata": {}, "id": hashlib.sha1(text.encode()).hexdigest()[:12],
              "source": text.splitlines(keepends=True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


notebook["cells"][0] = cell("markdown", """
# Objectness in few-shot video detection scores

Use a GPU for frozen encoder feature extraction and persistent storage for outputs. The temporal heads train on CPU. Both evaluation methods share the SAME three trained heads. The modification is `score = class_score * objectness ** beta` before the existing class-wise NMS. No additional training or trainable parameter is used for the proposed score.

This notebook prepares the original data and caches, trains the previously selected fixed reference, checks beta in {0, 0.5, 1} on validation, and stops unless the predeclared validation gate passes. If it passes, it prepares six untouched classes, fixes the selection, caches their features, records hashes, evaluates both methods, verifies AP, and exports uncertainty/results/shared weights. Earlier failed studies do not need to be rerun; their published manifests supply exclusion IDs. See `docs/SCORE_CALIBRATION_PLAN.md` for the exact protocol and limits. Do not choose settings from test results.
""")
notebook["cells"].append(cell("markdown", """
## Train the fixed reference and run the bounded score study

The cell streams progress and reuses verified completed work. It stops on errors. Keep the storage mounted. A complete run can take time when new encoder features are needed. Bootstrap analysis uses query videos, not independent frames.
"""))
cache_index = next(i for i, c in enumerate(notebook["cells"])
                   if c["cell_type"] == "code" and 'CACHE = EXPERIMENT_DIR / "cache"' in "".join(c["source"]))
notebook["cells"][cache_index] = cell("code", """
from adaptive_fsvod.cache import FrozenEncoder, build_cache
from adaptive_fsvod.data import VideoManifest, checksum, frame_key, read_json

CACHE = EXPERIMENT_DIR / "cache"
encoder = None
for split in ["train", "val", "test"]:
    manifest = VideoManifest(DATA / f"{split}.json")
    directory = CACHE / split
    metadata = read_json(directory / "metadata.json") if (directory / "metadata.json").exists() else {}
    expected = dict(manifest_sha256=checksum(manifest.path), encoder="google/owlv2-base-patch16-ensemble",
                    revision="cfd3195ba4ea9592eec887ded089f4c08eff231d", candidates=128,
                    support_per_class=16, pool_seed=42, synthetic=False)
    ready = metadata.get("complete") and all(metadata.get(k) == v for k, v in expected.items())
    if ready:
        pool = read_json(directory / "support_pool.json")
        needed = [directory / "frames" / f"{frame_key(vid, frame)}.pt"
                  for vid, video in manifest.videos.items() for frame in video["frames"]]
        needed += [directory / "support" / f"{item['key']}.pt" for values in pool.values() for item in values]
        ready = all(path.is_file() for path in needed)
    if ready:
        print(split, "complete cache reused without changing metadata")
    else:
        if encoder is None:
            encoder = FrozenEncoder(device="cuda", revision=expected["revision"], candidates=128)
        print(split, build_cache(manifest, directory, encoder, per_class=16))
if encoder is not None:
    del encoder
torch.cuda.empty_cache()
""")
notebook["cells"].append(cell("code", """
import json

def run_script(script, *arguments):
    command = [sys.executable, "-u", str(REPO / "scripts" / script), *map(str, arguments)]
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, bufsize=1) as process:
        for line in process.stdout:
            print(line, end="", flush=True)
        status = process.wait()
        if status:
            raise subprocess.CalledProcessError(status, command)

RESULTS = EXPERIMENT_DIR / "exports" / "score_calibration_results"
PROTOCOL = RESULTS / "confirmation_protocol"
run_script("train_fixed_reference.py", "--experiment-dir", EXPERIMENT_DIR, "--threads", 4)
for stage in ["freeze", "validate", "export"]:
    run_script("run_score_calibration.py", "--experiment-dir", EXPERIMENT_DIR,
               "--stage", stage, "--output", RESULTS)
decision = json.loads((EXPERIMENT_DIR / "score_calibration" / "validation_results.json").read_text())
if decision["validation_gate_passed"]:
    for stage in ["prepare", "download", "cache", "freeze-cache"]:
        run_script("prepare_score_confirmation.py", "--experiment-dir", EXPERIMENT_DIR,
                   "--stage", stage, "--output", PROTOCOL,
                   "--device", "cuda" if torch.cuda.is_available() else "cpu", "--threads", 4)
    run_script("evaluate_score_confirmation.py", "--experiment-dir", EXPERIMENT_DIR, "--threads", 4)
    run_script("analyse_score_confirmation.py", "--experiment-dir", EXPERIMENT_DIR, "--output", RESULTS)
else:
    print("Validation gate failed. Stop this family. No new test data were scored.")
run_script("export_score_models.py", "--experiment-dir", EXPERIMENT_DIR)
"""))
notebook["cells"].append(cell("code", """
from IPython.display import display
from PIL import Image
comparison = RESULTS / "comparison.csv"
print(comparison.read_text() if comparison.exists() else (RESULTS / "validation.csv").read_text())
figure = RESULTS / "score_calibration.png"
if figure.exists():
    display(Image.open(figure))
print("Shared inference heads:", EXPERIMENT_DIR / "exports" / "score_calibration_inference.zip")
print("Use --objectness-beta 0 for baseline and --objectness-beta 1 for the selected score.")
"""))
destination = root / "notebooks/score_calibration.ipynb"
destination.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
print(destination)
