# Runbook

Commands below are for a Linux Colab or Kaggle GPU session. The notebook provides the same steps. Run from the repository root. Enable internet access in Kaggle for data and model downloads.

## Install

```bash
git clone https://github.com/hskad/adaptive-fsvod.git
cd adaptive-fsvod
python -m pip install -r requirements.txt
python -m pytest -q
```

Python 3.10 or later and PyTorch 2.2 or later are required. Transformers is pinned to 4.46.3 because extraction uses its OWLv2 internals. Keep the runtime's installed PyTorch/CUDA pair. Do not replace CUDA wheels without a reason. Record actual versions with the run configuration. No particular free GPU type or availability is assumed.

For local Windows checks:

```powershell
python -m venv --system-site-packages .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q
```

## Software-only check

```bash
python -m adaptive_fsvod.smoke --output runs/synthetic_check --steps 4
```

This checks both methods, gradients, checkpoint loading, and COCO evaluation with synthetic inputs. Its metrics and weights must not be presented as research results. Select another output directory for a repeated check.

## Cache real data

Complete [data preparation](DATA.md) first. Its downloader retrieves and verifies only the selected RGB frames from a pinned public archive. The native OWLv2 image processing uses a 960-pixel padded square. The top 128 objectness tokens are kept after encoder inference. This reduces head training cost; it does not reduce the encoder's own image size or extraction cost.

Optional real-data integration check before constructing the full cache:

```bash
python scripts/check_real_episode.py --manifest data/pilot/train.json --output runs/real_episode_check.json --device cuda
```

This runs one update for each method on one real training episode. It is not an overfit experiment or an accuracy result.

```bash
python -m adaptive_fsvod.cache --manifest data/pilot/train.json --output cache/pilot/train --device cuda --revision cfd3195ba4ea9592eec887ded089f4c08eff231d --support-per-class 16
python -m adaptive_fsvod.cache --manifest data/pilot/val.json --output cache/pilot/val --device cuda --revision cfd3195ba4ea9592eec887ded089f4c08eff231d --support-per-class 16
python -m adaptive_fsvod.cache --manifest data/pilot/test.json --output cache/pilot/test --device cuda --revision cfd3195ba4ea9592eec887ded089f4c08eff231d --support-per-class 16
```

The public encoder weights download automatically from `google/owlv2-base-patch16-ensemble` on Hugging Face. The immutable revision is recorded in cache metadata. A cache can resume after interruption. A cache with different settings requires a different directory.

## Fix episodes

Start with 2-way 1-shot episodes. Training uses four consecutive selected frames. Validation and test use full selected query sequences. Support crops come from other videos. Episode files are shared by both methods.

```bash
python -m adaptive_fsvod.episodes --manifest data/pilot/train.json --cache cache/pilot/train --output data/pilot/train_episodes.json --episodes 300 --n-way 2 --k-shot 1 --max-frames 4 --seed 42
python -m adaptive_fsvod.episodes --manifest data/pilot/val.json --cache cache/pilot/val --output data/pilot/val_episodes.json --episodes 20 --n-way 2 --k-shot 1 --seed 43
python -m adaptive_fsvod.episodes --manifest data/pilot/test.json --cache cache/pilot/test --output data/pilot/test_episodes.json --episodes 30 --n-way 2 --k-shot 1 --seed 44
```

## Train paired methods

Initial shared budget: 300 episode updates, AdamW, learning rate 1e-4, weight decay 0.01, 5% linear warmup, cosine decay, gradient norm cap 0.1. The loss uses background-weighted cross entropy and Hungarian-matched L1 plus generalized IoU. Classification and box weights are 2 and 5. These choices are shared adaptations, not the contribution.

```bash
python -m adaptive_fsvod.train --train-manifest data/pilot/train.json --val-manifest data/pilot/val.json --train-cache cache/pilot/train --val-cache cache/pilot/val --train-episodes data/pilot/train_episodes.json --val-episodes data/pilot/val_episodes.json --config configs/fixed.json --output runs/fixed_seed42 --steps 300 --eval-every 50 --seed 42 --device cuda
python -m adaptive_fsvod.train --train-manifest data/pilot/train.json --val-manifest data/pilot/val.json --train-cache cache/pilot/train --val-cache cache/pilot/val --train-episodes data/pilot/train_episodes.json --val-episodes data/pilot/val_episodes.json --config configs/adaptive.json --output runs/adaptive_seed42 --steps 300 --eval-every 50 --seed 42 --device cuda
```

The encoder is not part of either optimizer. The two configurations differ only in `memory_mode`. The initial 0.7 fixed threshold and adaptive settings are starting values, not validated best settings. Give both methods a small validation search before final test evaluation. Use the same search budget. Keep test data out of this search. If tuning the fixed threshold, change a copy of `configs/fixed.json` and give it a unique run directory.

To resume, repeat the exact training command with `--resume runs/fixed_seed42/last.pt`. Keep the total steps, seed, learning rate, configuration, and episodes unchanged. Checkpoints are saved at each validation interval. The most recent incomplete interval is rerun after a disconnect.

Run a short overfit experiment on a few training episodes before the full budget. A finite loss alone does not establish useful learning. Inspect loss trends and detections. Adjust shared implementation problems for both methods. Record all changes in Git.

## Evaluate

Select `best.pt` using validation AP. Evaluate the two selected runs once on the fixed test episodes.

```bash
python -m adaptive_fsvod.evaluate --manifest data/pilot/test.json --cache cache/pilot/test --episodes data/pilot/test_episodes.json --checkpoint runs/fixed_seed42/best.pt --output runs/fixed_seed42/test --device cuda
python -m adaptive_fsvod.evaluate --manifest data/pilot/test.json --cache cache/pilot/test --episodes data/pilot/test_episodes.json --checkpoint runs/adaptive_seed42/best.pt --output runs/adaptive_seed42/test --device cuda
python -m adaptive_fsvod.compare --fixed runs/fixed_seed42/test/metrics.json --adaptive runs/adaptive_seed42/test/metrics.json --output runs/comparison_seed42
```

AP is frame-level COCO box AP over IoU 0.50:0.05:0.95, with at most 100 detections per frame. AP50 and AP75 are also saved. Every episode/frame has a unique image ID. Empty frames remain in evaluation. Query objects outside episode classes count as background. A validation/test episode is a repeated sampling unit; do not treat every frame as an independent statistical trial.

The initial reduced protocol does not sample all possible episodes or reproduce the original benchmark. Test classes are unseen during our head training; they can have appeared in public encoder pretraining. Report these limits.

`cached_pipeline_fps` excludes the image encoder. Label it accordingly. It cannot be compared directly with the paper's end-to-end FPS. Use `infer` for an end-to-end frame timing sample. Use more paired training seeds if time permits; report their variability.

## Automated pilot

After manifests, caches, and episode files exist, the runner can execute the predefined paired training matrix, resume saved checkpoints with their original budgets, evaluate best checkpoints, build paired comparisons, print validation diagnostics and aggregate AP, and export evidence with a SHA-256 manifest:

```bash
python -u -m adaptive_fsvod.pilot --experiment-dir /content/drive/MyDrive/adaptive-fsvod-experiment --action run --seeds 42 43 44 --device cuda
```

The defaults are 300 steps, validation every 50 steps, and repository `fixed.json`/`adaptive.json` configurations. Completed training runs have no remaining optimizer steps; resume checks still verify their original provenance. Evaluation is repeated in `run` mode. No automatic threshold search or test-based method selection occurs. Source data and cached features are reused.

For already completed runs, this command reads their results, prints validation diagnostics, creates comparisons and aggregate CSV/JSON, and archives all specified seeds without training or GPU evaluation:

```bash
python -u -m adaptive_fsvod.pilot --experiment-dir /content/drive/MyDrive/adaptive-fsvod-experiment --action collect --seeds 42 43 44
```

Outputs are `exports/pilot_summary.csv`, `exports/pilot_summary.json`, `exports/pilot_seeds_42_43_44.zip`, its `.manifest.json` checksum record, and `runs/automation_status.json`. The archive contains the selected run directories, data manifests and episode files, cache metadata and support pools, comparisons, and the overfit check if present. Source images, feature tensors, encoder weights, and existing archives are excluded. Ensure enough persistent storage for a second copy of the trained head checkpoints. A failed command records its phase and error and stops; it does not silently skip a model or seed. Fix the error and repeat the same command.

This automates execution within the runtime. Browser access or a shared artifact is still needed for a collaborator to inspect the output remotely.

## Inference

Put ordered frames from one video in `demo/frames/`. Put cropped support objects in `demo/support/<class-name>/`. Folder names define the classes. Use zero-padded frame names.

```bash
python -m adaptive_fsvod.infer --frames demo/frames --support demo/support --checkpoint runs/adaptive_seed42/best.pt --output runs/demo_adaptive --device cuda
```

Outputs are annotated PNGs, `detections.json`, and `timing.json`. Memory is reset when the command starts. The command rejects synthetic checkpoints.

## Preserve the run

Save caches, manifests, episodes, checkpoints, and logs to persistent storage before a free session ends. The notebook lets you set a persistent experiment directory. Publish real `best.pt` files through GitHub Releases or another public download location. Include the matching config, data manifest, episode files, `training.jsonl`, evaluation JSON, Git commit, environment versions, and SHA-256 for each weight file. Do not commit datasets or weights directly to Git.

The GPU-session pilot has trained real-video checkpoints and measured results for seeds 42/43/44; see [the result record](PILOT_RESULTS.md). Public checkpoint links and imported source artifacts are still pending. Add links after a fresh download has been checked.
