# Adaptive FSVOD

[![Software checks](https://github.com/hskad/adaptive-fsvod/actions/workflows/checks.yml/badge.svg)](https://github.com/hskad/adaptive-fsvod/actions/workflows/checks.yml)

**Objectness in the final detection score improves a small few-shot video detection pipeline from 47.44 to 58.30 AP.** This course project starts from [Temporal Object-Aware Vision Transformer for Few-Shot Video Object Detection](https://ojs.aaai.org/index.php/AAAI/article/view/37497), Yogesh Kumar and Anand Mishra, AAAI 2026, and the [authors' implementation](https://github.com/yogesh-iitj/fs-video-vit).

We use a documented smaller adaptation with a frozen OWLv2 encoder and trained temporal heads. We do not reproduce the paper's full model or benchmark scores. Our contribution is the implementation and controlled test of an objectness score in this reduced pipeline. Objectness scoring is an established detection idea; we do not claim to have invented it. Earlier unsuccessful experiments remain available.

## Method and results

For candidate j, the baseline ranks detections by foreground class confidence p_j. Our change uses **`s_j = p_j * q_j^beta`**, where q_j is the frozen encoder's objectness for the same candidate. Validation selected beta=1 from {0, 0.5, 1}. We apply the product before the existing score filter and class-wise NMS. Foreground/background decisions, boxes, temporal memory, encoder, and trained weights stay fixed. Both methods use the **same three checkpoints**; the proposed score adds no training, encoder pass, or trainable parameter.

For positive scores, `log(s_j) = log(p_j) + beta * log(q_j)`: proposals with weak evidence of an object receive a lower rank. This is a ranking heuristic, not a calibrated joint probability. See [the study plan](docs/SCORE_CALIBRATION_PLAN.md) and [modified decoder](src/adaptive_fsvod/evaluate.py).

| Method | AP mean +/- seed SD | AP50 | AP75 |
|---|---:|---:|---:|
| Baseline, beta=0 | 47.440 +/- 0.091 | 62.435 | 50.852 |
| Objectness score, beta=1 | **58.298 +/- 0.182** | **75.044** | **62.798** |

The mean paired gain is **10.858 AP points**. All three training-seed pairs and all six class means improved. The 1,000-draw paired, class-stratified query-video bootstrap interval is **[+5.960, +14.850]** AP points. It is conditional on the fixed classes, support episodes, and trained seed pairs. Bear supplies most of the mean gain. These results do not establish the same gain on other classes or the paper's full benchmarks.

The fresh confirmation has six previously unused classes, 24 videos, 192 distinct frames, and 120 two-way one-shot episodes (960 episode/frame instances). Metrics are frame-level COCO **box** AP at IoU 0.50:0.05:0.95, AP50, and AP75, maxDets=100. Models, settings, data, episodes, and shared features were frozen before scoring. [Raw results, class effects, uncertainty, and checksums](results/score_calibration/README.md).

## Install and reproduce

Use Python 3.12. The measured final environment used PyTorch 2.9.1+cpu, Transformers 4.46.3, NumPy 1.26.4, SciPy 1.13.0, Pillow 10.3.0, and pycocotools 2.0.11 on Windows 11, Intel i5-1340P, 16 GB RAM, four CPU threads. **No CUDA is required for saved-feature evaluation or shared-head training.** Initial pilot features used a Tesla T4, PyTorch 2.11.0+cu130, CUDA 13.0; final confirmation features used CPU SDPA. Each run includes its recorded environment.

```bash
git clone https://github.com/hskad/adaptive-fsvod.git
cd adaptive-fsvod
python -m venv .venv
```

Activate with `source .venv/bin/activate` on Linux or `.venv\Scripts\Activate.ps1` in PowerShell, then:

```bash
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-reproduce.txt
python scripts/download_artifacts.py --experiment-dir ../fsvod-reproduction
python scripts/reproduce_results.py --experiment-dir ../fsvod-reproduction --threads 4
```

The downloader verifies archive/file SHA-256 and prepares the measured features, fixed episodes, model files, and selected configuration. The last command evaluates all three heads with both scores, saves metrics/predictions, and checks them against the table. Outputs: `../fsvod-reproduction/reproduced/`; summary: `reproduced/summary.json`.

**[Complete evaluator instructions](docs/REPRODUCE.md)** include Windows/Linux installation, directory layout, raw-image preparation, direct training/evaluation/inference commands, resume, and expected outputs. For Colab/Kaggle, retain the matched PyTorch/CUDA installation and use `requirements.txt`. The [full training notebook](https://colab.research.google.com/github/hskad/adaptive-fsvod/blob/main/notebooks/score_calibration.ipynb) prepares data and runs the study; its encoder cells use a GPU.

## Dataset and preprocessing

Data are a custom FSYTV-40 subset derived from YouTube-VIS 2019. Head training/validation use 6/3 classes, 18/9 videos, and 131/50 frames. The final six-class confirmation is disjoint from training, validation, and all earlier tests. It is not the paper's standard split. Public encoder pretraining may include test object classes.

- [FSYTV annotations](https://drive.google.com/drive/folders/1a1PpfAxeYL7AbxYViDDnx7ACFtRohVL5), linked by [FewX](https://github.com/fanq15/FewX).
- RGB archive: [JiaHuang01/Youtubevos_train](https://huggingface.co/datasets/JiaHuang01/Youtubevos_train), revision `7855adf7e2e8e9386b578f58d490f25c1886368d`.
- [Original provider](https://youtube-vos.org/dataset/vis/) and [research-use terms](https://youtube-vos.org/dataset/term/).
- [Download/preparation commands, checksums, and image layout](docs/DATA.md); [exact final feature rebuild](docs/REPRODUCE.md#5-prepare-rgb-data-and-rebuild-features).

The downloader retrieves selected ZIP ranges and checks CRC, dimensions, decoding, and SHA-256. Native 960-pixel padded-square processing supplies the top 128 candidate features (dimension 512). Query features are float16; support features are float32. Support crops come from other videos. Preserve frozen manifest bytes; use `--images-root` to relocate images. RGB images and public encoder weights are not in the repository or release bundles.

## Train baseline and proposed method

After the artifact download above, reproduce the three shared heads:

```bash
python scripts/train_fixed_reference.py --experiment-dir ../fsvod-reproduction --threads 4
```

This trains seeds 42/43/44 for 300 episode updates each with the [selected configuration](results/revision_v2/configs/fixed_tau060.json), validates every 50 updates, and selects `best.pt` by validation AP. Settings: AdamW, learning rate 1e-4, weight decay 0.01, 5% warmup, cosine decay, gradient cap 0.1, classification/box loss weights 2/5, Hungarian L1+GIoU matching, memory threshold 0.6, and 16-token cap per class. The frozen encoder uses pinned public [weights](https://huggingface.co/google/owlv2-base-patch16-ensemble/tree/cfd3195ba4ea9592eec887ded089f4c08eff231d). The heads have 2,103,815 trainable parameters.

The proposed method uses these heads unchanged. Set beta=0 for baseline and beta=1 for the proposed score at evaluation. No second training run is needed. [Direct training commands and resume](docs/REPRODUCE.md#3-reproduce-shared-head-training).

## Checkpoints, configurations, logs, and evaluation

**[Public release v1.0-results](https://github.com/hskad/adaptive-fsvod/releases/tag/v1.0-results)**:

- [Trained heads and score presets, 23.4 MB](https://github.com/hskad/adaptive-fsvod/releases/download/v1.0-results/score_calibration_inference.zip): seeds 42/43/44, selected steps 250/250/300, shared by both methods. [SHA-256 manifest](results/score_calibration/inference_bundle.manifest.json).
- [Measured train/validation/confirmation features and episodes, 47.3 MB](https://github.com/hskad/adaptive-fsvod/releases/download/v1.0-results/score_calibration_features.zip). [SHA-256 manifest](results/score_calibration/features_bundle.manifest.json). Derived features retain dataset research-use terms.
- Training configurations/logs: [seed 42](results/revision_v2/source_runs/fixed_tau060_seed42/), [43](results/revision_v2/source_runs/fixed_tau060_seed43/), [44](results/revision_v2/source_runs/fixed_tau060_seed44/).
- [Validation search](results/score_calibration/validation.csv), [six final evaluations](results/score_calibration/evaluations/), [mean comparison](results/score_calibration/comparison.csv), [per-class results](results/score_calibration/per_class.csv), [bootstrap records](results/score_calibration/bootstrap_draws.json), [frozen data/episodes/cache](results/score_calibration/confirmation_protocol/).

Inference exports omit optimizer/RNG state. Fresh training writes full resumable `best.pt` and `last.pt`. No separate proposed training log exists because the score needs no extra training.

`python -m adaptive_fsvod.evaluate` accepts the manifest, cache, episodes, checkpoint, score exponent, output path, and device. `python -m adaptive_fsvod.infer` processes ordered frames and support crops, and writes annotated frames, `detections.json`, and `timing.json`. [Exact executable commands](docs/REPRODUCE.md#4-evaluate-or-infer-with-an-individual-checkpoint). Cached FPS excludes the encoder; no end-to-end speed gain is claimed.

## Code organization and verification

```text
src/adaptive_fsvod/
  convert_fsytv.py, prepare.py, download_frames.py  Annotation/subset/image preparation
  data.py, episodes.py                            Manifest and support/query checks
  cache.py, encoder_attention.py                  Frozen features, optional SDPA
  model.py, loss.py, train.py                     Temporal detector and training
  evaluate.py                                    Proposed score in decode(); COCO AP
  infer.py                                       Raw-frame inference with score flag
  bootstrap.py                                   Paired video resampling
scripts/
  download_artifacts.py, reproduce_results.py     Public CPU reproduction
  train_fixed_reference.py                        Three shared reference heads
  run_score_calibration.py                        Frozen validation selection
  prepare_score_confirmation.py                   New-class data and feature freeze
  evaluate_score_confirmation.py                  Paired final evaluation
  analyse_score_confirmation.py                   Verification, uncertainty, plots
  export_score_models.py                         Shared inference head export
  export_reproduction_features.py                 Measured feature export
notebooks/score_calibration.ipynb                  Full cloud preparation/training
results/score_calibration/                        Final evidence and protocols
tests/                                           Software and metric checks
docs/                                            Data, reproduction, experiment plans
```

```bash
python -m pytest -q
python -m adaptive_fsvod.smoke --output ../fsvod-reproduction/synthetic_check --steps 4
```

Synthetic scores check software only. A [separate clean reproduction](results/reproducibility_20261025/README.md) retrained all three heads to exactly equal model tensors and reproduced all six detection records, all 18 AP/AP50/AP75 values, and all 1,000 bootstrap draws exactly. [Validation records](docs/VALIDATION.md) state tested paths and platform limits. Commit history, versions, configuration/data hashes, and seeds preserve provenance. Weights, images, and caches stay out of Git; release assets supply measured artifacts. The [course requirement checklist](docs/SUBMISSION_READINESS.md) maps every non-report requirement to its files.

## Report draft

The [one-page LaTeX report source](report/README.md) uses the instructor's template with its preamble unchanged. It contains the paper summary, exact modification, mathematical justification, and measured comparison. The five member names and roll numbers must be added before submission. The reviewed draft fits one A4 page.

## Earlier studies

[Adaptive threshold](results/pilot_20261024/README.md), [small centering test](results/revision_v5/README.md), and [larger centering check](results/larger_confirmation/README.md) records remain available, including negative results. The historical [runbook](docs/RUNBOOK.md) contains their commands.
