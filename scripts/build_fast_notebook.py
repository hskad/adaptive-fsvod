"""Reuse verified setup cells and run the current study with one command."""

import copy
import hashlib
import json
from pathlib import Path


root = Path(__file__).resolve().parents[1]
source = json.loads((root / "notebooks/colab_kaggle_pilot.ipynb").read_text(encoding="utf-8"))
index = next(i for i, cell in enumerate(source["cells"])
             if cell["cell_type"] == "code" and "for split, count, episode_seed" in "".join(cell["source"]))
notebook = copy.deepcopy(source)
notebook["cells"] = notebook["cells"][:index + 1]


def cell(kind, text):
    text = text.strip() + "\n"
    value = {"cell_type": kind, "metadata": {}, "id": hashlib.sha1(text.encode()).hexdigest()[:12],
             "source": text.splitlines(keepends=True)}
    if kind == "code":
        value.update({"execution_count": None, "outputs": []})
    return value


notebook["cells"][0] = cell("markdown", """
# Feature centering for fast few-shot video adaptation

Run setup, data preparation, feature caching, and episode generation below. Mount Drive and choose persistent storage before caching. A GPU is recommended for the initial frozen encoder cache. The final paired heads use CPU.

The current method subtracts a class-balanced mean fitted only on training support features. Both methods execute exactly **50 optimizer updates** with the first 50 updates of a common 300-update learning-rate schedule. The available training episode file contains 300 episodes; only the first 50 are consumed.

Historical small-set AP: fixed 51.43 ± 0.93, centered 52.04 ± 0.08 across three training seeds. It contains 3 new classes, 6 videos, and 18 distinct frames. A later frozen-model check on 6 further unseen classes, 24 videos, and 192 frames gave fixed/centered AP 38.61/35.14. All three centered runs were worse. The larger check did not confirm faster adaptation. See `results/larger_confirmation/README.md` and `results/revision_v5/README.md` for both results. This notebook reproduces the historical limited-update study. Earlier failed experiments remain available.
""")
notebook["cells"].append(cell("markdown", """
## Run the current experiment

The cell below resumes existing checkpoints, runs an equal-budget validation search and three-seed comparison, prepares the fixed fresh subset, caches it, freezes input hashes, evaluates, verifies all metrics and inference weights, and exports a CSV, figure, and checkpoint ZIP. Confirmation encoder extraction uses CPU to match the recorded result; it can take several minutes. No manual copying between stages is needed. Errors stop execution.
"""))
notebook["cells"].append(cell("code", """
def run_script(script, *arguments):
    command = [sys.executable, "-u", str(REPO / "scripts" / script), *map(str, arguments)]
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, bufsize=1) as process:
        for line in process.stdout:
            print(line, end="", flush=True)
        status = process.wait()
        if status:
            raise subprocess.CalledProcessError(status, command)

run_script("run_fast_study.py", "--experiment-dir", EXPERIMENT_DIR,
           "--stage", "all", "--threads", 4)
run_script("export_fast_results.py", "--experiment-dir", EXPERIMENT_DIR,
           "--output", EXPERIMENT_DIR / "exports" / "verified_fast_results")
"""))
notebook["cells"].append(cell("code", """
from IPython.display import display
from PIL import Image
RESULTS = EXPERIMENT_DIR / "exports" / "verified_fast_results"
print((RESULTS / "comparison.csv").read_text())
display(Image.open(RESULTS / "fast_adaptation.png"))
print("Inference weights:", EXPERIMENT_DIR / "exports" / "fast_adaptation_inference.zip")
"""))
destination = root / "notebooks/fast_adaptation.ipynb"
destination.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
print(destination)
