# Adaptive FSVOD

A course project on feature calibration and temporal memory for few-shot video object detection.

The starting point is [Temporal Object-Aware Vision Transformer for Few-Shot Video Object Detection](https://ojs.aaai.org/index.php/AAAI/article/view/37497) by Yogesh Kumar and Anand Mishra (AAAI 2026). The [authors' implementation](https://github.com/yogesh-iitj/fs-video-vit) is the reference.

This project uses a smaller, documented adaptation. It does not reproduce the paper's full model or claim its published scores. The image encoder is frozen. A small temporal decoder and detection heads are trained. The current contribution is class-balanced training-feature centering for limited-update adaptation. Earlier memory-rule proposals remain as negative experiments.

## Status

**Current result: the larger test did not confirm the centering gain.** On six further unseen classes, 24 videos, and 192 distinct frames, the frozen 50-update fixed/centered models averaged **38.61/35.14 AP** (seed SD **0.29/0.13**). Centering was lower in all three seed pairs, by **3.47 AP points on average**. The separately tuned 300-update baseline reference averaged **55.10 AP**. [Larger confirmation, uncertainty analysis, and source records](results/larger_confirmation/README.md).

The larger set's paired, class-stratified video-bootstrap 95% interval was **[-7.29, +0.20] AP points**, conditional on these classes, support episodes, and trained seeds. It includes zero: this check establishes neither a significant gain nor a significant loss.

The earlier three-class, six-video, 18-frame subset gave fixed/centered AP **51.43/52.04**, with two positive pairs. [That small-set result](results/revision_v5/README.md) remains available, but it does not support a general faster-adaptation claim after this larger check. All nine inference heads are verified and bundled locally; public weight links remain pending.

The first software milestone is implemented:

- Data conversion, validated manifests, and deterministic support/query episodes.
- Frozen OWLv2 feature extraction with resumable caches.
- Fixed and adaptive temporal memory rules with the same trainable heads.
- Training, validation checkpoint selection, resume, inference, COCO box evaluation, and paired result plots.
- A Colab/Kaggle notebook and CPU checks in GitHub Actions.

Local tests and synthetic end-to-end training passed. Public pretrained extraction and a real-data training update for both methods also passed. See [validation records](docs/VALIDATION.md).

**The first real-video pilot is complete for three training seeds.** All 249 selected RGB frames passed verification. On the 30 fixed test episodes, fixed/adaptive AP averaged 45.64/45.34, with sample standard deviations 0.32/0.22. Adaptive test AP was lower for every seed. [Source logs, configurations, predictions, and locally verified results](results/pilot_20261024/README.md) are now included. Checkpoints are verified locally; public download links remain pending. Synthetic outputs and the single-update integration check are not research results.

## Start

Use the setup notebook through feature caching and episode generation, then run the current study:

```bash
python -u scripts/run_fast_study.py --experiment-dir /path/to/adaptive-fsvod-experiment --stage all --threads 4
python scripts/export_fast_results.py --experiment-dir /path/to/adaptive-fsvod-experiment --output results/revision_v5
```

This searches and repeats both 50-update methods, prepares the fresh confirmation subset, caches its features, freezes inputs, and evaluates. It runs on CPU by default. The existing notebook's fixed/adaptive training section and `pilot` runner below reproduce the historical threshold experiment. [Current method, results, and exact budget](results/revision_v5/README.md).

[Open the current notebook in Colab](https://colab.research.google.com/github/hskad/adaptive-fsvod/blob/main/notebooks/fast_adaptation.ipynb), or upload [the notebook](notebooks/fast_adaptation.ipynb) to Kaggle. Enable a GPU for the initial encoder cache and select persistent storage. It prepares data and runs the current study automatically. The [historical notebook](notebooks/colab_kaggle_pilot.ipynb) reproduces the earlier threshold pilot.

For local software checks:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python -m adaptive_fsvod.smoke --output runs/synthetic_check --steps 4
```

The [runbook](docs/RUNBOOK.md) contains exact preparation, training, inference, and evaluation commands. The [data guide](docs/DATA.md) records sources, checksums, the directory layout, and the image access issue.

After preparation and caching, one command can resume the paired training matrix, evaluate, summarize, and archive it:

```bash
python -u -m adaptive_fsvod.pilot --experiment-dir /path/to/adaptive-fsvod-experiment --action run --seeds 42 43 44 --device cuda
```

Use `--action collect` to summarize and archive completed runs without training or GPU evaluation. See the [automation instructions](docs/RUNBOOK.md#automated-pilot).

## Experiment

- Encoder: `google/owlv2-base-patch16-ensemble`.
- Baseline: a fixed threshold selects object features for the next frame.
- Tested modification: class-balanced training-feature centering, intended to improve limited-update adaptation; see `configs/fast_adaptation/`. The larger check did not confirm a gain.
- Historical proposal: a bounded, class-dependent confidence threshold. It did not improve the initial test scores.
- Controls: the same cached features, episodes, model size, token cap, optimizer, and evaluation procedure.
- Data: a verified custom FSYTV-40 subset from YouTube-VIS 2019. It has 6/3/3 classes, 18/9/9 videos, and 131/50/68 frames for train/validation/test. Matching RGB frames come from a pinned YouTube-VOS archive mirror.
- Metrics: frame-level COCO box AP, AP50, AP75, and measured run time. This is not video mask AP.

See [the experiment plan](docs/EXPERIMENT_PLAN.md) for the scope and evaluation rules.

## Code organization

```text
configs/                  Paired fixed/adaptive model configurations
notebooks/                Colab/Kaggle pilot
src/adaptive_fsvod/
  convert_fsytv.py         Streaming annotation conversion
  download_frames.py       Verified partial RGB downloads from a pinned archive
  prepare.py              Class and video split preparation
  data.py, episodes.py     Manifest checks and fixed episodes
  cache.py                Frozen encoder and feature caches
  model.py                Shared detector, memory rules, and optional feature centering
  loss.py                 Hungarian matching and detection losses
  train.py                Training, validation, checkpoints, and resume
  evaluate.py, compare.py  Frame-level box AP and paired result files
  pilot.py                Resume, evaluate, summarize, and archive a paired pilot
  infer.py                Ordered-frame inference and timing
  smoke.py                Clearly labeled synthetic software check
tests/                    Data, memory, gradient, coordinate, and metric checks
docs/                     Experiment plan, data guide, runbook, and validation
```

## Checkpoints and results

The frozen public encoder is downloaded from [Hugging Face](https://huggingface.co/google/owlv2-base-patch16-ensemble), revision `cfd3195ba4ea9592eec887ded089f4c08eff231d`. Our temporal heads require training.

Real baseline and adaptive checkpoints: **trained and locally checksum-verified; public links pending**. [Complete source result records](results/pilot_20261024/README.md) include evaluation JSON, matching configurations, logs, checksums, and episode manifests. Saved test detections reproduce all six reported metric sets exactly. Free runtime outputs need persistent storage.

## Version control

Use small commits for setup, data preparation, the baseline, the adaptive rule, and evaluation. Record the Git commit, configuration, seed, package versions, and data manifest for each experiment. Keep datasets, credentials, caches, and model weights out of Git. Publish research checkpoints through an accessible download location when they exist.
