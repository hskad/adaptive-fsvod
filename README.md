# Adaptive FSVOD

A course project on feature calibration and temporal memory for few-shot video object detection.

The starting point is [Temporal Object-Aware Vision Transformer for Few-Shot Video Object Detection](https://ojs.aaai.org/index.php/AAAI/article/view/37497) by Yogesh Kumar and Anand Mishra (AAAI 2026). The [authors' implementation](https://github.com/yogesh-iitj/fs-video-vit) is the reference.

This project uses a smaller, documented adaptation. It does not reproduce the paper's full model or claim its published scores. The image encoder is frozen. A small temporal decoder and detection heads are trained. The current extension combines class confidence with the frozen encoder's objectness when ranking detections. Earlier memory and centering proposals remain as unsuccessful experiments.

## Status

**Current result: objectness in the final score improved the separate confirmation AP.** On six unused classes, 24 videos, and 192 distinct frames, baseline/objectness AP averaged **47.44/58.30** (seed SD **0.09/0.18**). The mean paired gain was **10.86 AP points**, with all three training-seed pairs positive. The paired, class-stratified video-bootstrap 95% interval was **[+5.96, +14.85]**, conditional on these classes, support episodes, and fixed trained seed pairs. [Verified results, source predictions, class differences, and limits](results/score_calibration/README.md).

Both methods use the same three validation-selected fixed heads with a 300-update training budget. The new rule is `score = class_score * objectness`; it adds zero training updates and no encoder pass or trainable parameter. Validation selected exponent 1 from {0, 0.5, 1} before the unused classes were scored. This applies a known detection idea to this reduced pipeline; no global novelty or full-paper benchmark gain is claimed. All six classes improved, but the bear class accounts for most of the mean gain. Shared head tensors and decoder presets are verified and bundled locally; public weight links remain pending.

The earlier centering study had a small six-video gain that failed its larger check: fixed/centered 50-update AP was **38.61/35.14**, with a conditional 95% interval **[-7.29, +0.20]** for the difference. [Those records](results/larger_confirmation/README.md) and the [earlier small set](results/revision_v5/README.md) remain unchanged. The score study uses further unused classes, so its absolute AP must not be compared directly with that older set's AP.

The first software milestone is implemented:

- Data conversion, validated manifests, and deterministic support/query episodes.
- Frozen OWLv2 feature extraction with resumable caches.
- Fixed and adaptive temporal memory rules with the same trainable heads.
- Training, validation checkpoint selection, resume, inference, COCO box evaluation, and paired result plots.
- A Colab/Kaggle notebook and CPU checks in GitHub Actions.

Local tests and synthetic end-to-end training passed. Public pretrained extraction and a real-data training update for both methods also passed. See [validation records](docs/VALIDATION.md).

**The first real-video pilot is complete for three training seeds.** All 249 selected RGB frames passed verification. On the 30 fixed test episodes, fixed/adaptive AP averaged 45.64/45.34, with sample standard deviations 0.32/0.22. Adaptive test AP was lower for every seed. [Source logs, configurations, predictions, and locally verified results](results/pilot_20261024/README.md) are now included. Checkpoints are verified locally; public download links remain pending. Synthetic outputs and the single-update integration check are not research results.

## Start

Use the [score-study notebook in Colab](https://colab.research.google.com/github/hskad/adaptive-fsvod/blob/main/notebooks/score_calibration.ipynb), or upload [the notebook](notebooks/score_calibration.ipynb) to Kaggle. Enable a GPU for feature extraction and choose persistent storage. After the original data caches and episodes exist, the initial commands are:

```bash
python scripts/train_fixed_reference.py --experiment-dir /path/to/adaptive-fsvod-experiment --threads 4
python scripts/run_score_calibration.py --experiment-dir /path/to/adaptive-fsvod-experiment --stage freeze
python scripts/run_score_calibration.py --experiment-dir /path/to/adaptive-fsvod-experiment --stage validate --threads 4
```

The [runbook](docs/RUNBOOK.md#objectness-score-study) provides the remaining preparation, frozen confirmation, bootstrap verification, shared-head export, and evaluation-only commands. The notebook runs those stages automatically if the validation gate passes. Existing heads are kept and the proposed score needs no further training. CPU feature extraction is supported; it is slower than GPU extraction.

The [centering notebook](notebooks/fast_adaptation.ipynb) and [initial threshold notebook](notebooks/colab_kaggle_pilot.ipynb) reproduce the earlier experiments.

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
- Current modification: objectness in final detection scores; see [the fixed plan](docs/SCORE_CALIBRATION_PLAN.md). Use `--objectness-beta 0` for baseline or `--objectness-beta 1` for the selected score.
- Earlier modification: training-feature centering, whose larger check did not confirm a gain.
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
