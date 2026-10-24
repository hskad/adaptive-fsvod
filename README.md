# Adaptive FSVOD

A course project on adaptive temporal confidence filtering for few-shot video object detection.

The starting point is [Temporal Object-Aware Vision Transformer for Few-Shot Video Object Detection](https://ojs.aaai.org/index.php/AAAI/article/view/37497) by Yogesh Kumar and Anand Mishra (AAAI 2026). The [authors' implementation](https://github.com/yogesh-iitj/fs-video-vit) is the reference.

This project uses a smaller, documented adaptation. It does not reproduce the paper's full model or claim its published scores. The image encoder is frozen. A small temporal decoder and detection heads are trained. The controlled change is the rule that selects features for temporal memory.

## Status

The first software milestone is implemented:

- Data conversion, validated manifests, and deterministic support/query episodes.
- Frozen OWLv2 feature extraction with resumable caches.
- Fixed and adaptive temporal memory rules with the same trainable heads.
- Training, validation checkpoint selection, resume, inference, COCO box evaluation, and paired result plots.
- A Colab/Kaggle notebook and CPU checks in GitHub Actions.

Local tests and synthetic end-to-end training passed. Public pretrained extraction and a real-data training update for both methods also passed. See [validation records](docs/VALIDATION.md).

**The first real-video pilot is complete for three training seeds.** All 249 selected RGB frames passed verification. On the 30 fixed test episodes, fixed/adaptive AP averaged 45.64/45.34, with sample standard deviations 0.32/0.22. Adaptive test AP was lower for every seed. See [pilot results and limitations](docs/PILOT_RESULTS.md). Checkpoints were saved in the experiment Drive directory; public download links and source logs are still pending. Synthetic outputs and the single-update integration check are not research results.

## Start

[Open the GPU notebook in Colab](https://colab.research.google.com/github/hskad/adaptive-fsvod/blob/main/notebooks/colab_kaggle_pilot.ipynb), or upload [the notebook](notebooks/colab_kaggle_pilot.ipynb) to Kaggle. Enable a GPU and select persistent storage for outputs. The notebook prepares the subset and downloads its frames automatically.

For local software checks:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python -m adaptive_fsvod.smoke --output runs/synthetic_check --steps 4
```

The [runbook](docs/RUNBOOK.md) contains exact preparation, training, inference, and evaluation commands. The [data guide](docs/DATA.md) records sources, checksums, the directory layout, and the image access issue.

## Experiment

- Encoder: `google/owlv2-base-patch16-ensemble`.
- Baseline: a fixed threshold selects object features for the next frame.
- Proposed method: a bounded, class-dependent threshold uses the mean and standard deviation of current confidence scores.
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
  model.py                Shared detector; select_memory is the controlled change
  loss.py                 Hungarian matching and detection losses
  train.py                Training, validation, checkpoints, and resume
  evaluate.py, compare.py  Frame-level box AP and paired result files
  infer.py                Ordered-frame inference and timing
  smoke.py                Clearly labeled synthetic software check
tests/                    Data, memory, gradient, coordinate, and metric checks
docs/                     Experiment plan, data guide, runbook, and validation
```

## Checkpoints and results

The frozen public encoder is downloaded from [Hugging Face](https://huggingface.co/google/owlv2-base-patch16-ensemble), revision `cfd3195ba4ea9592eec887ded089f4c08eff231d`. Our temporal heads require training.

Real baseline and adaptive checkpoints: **trained in the GPU session; public links pending**. The [first comparison](docs/PILOT_RESULTS.md) records values supplied from that session. Publish the source evaluation JSON, matching configurations, logs, checksums, and episode manifests with the checkpoints. Free runtime outputs need persistent storage.

## Version control

Use small commits for setup, data preparation, the baseline, the adaptive rule, and evaluation. Record the Git commit, configuration, seed, package versions, and data manifest for each experiment. Keep datasets, credentials, caches, and model weights out of Git. Publish research checkpoints through an accessible download location when they exist.
