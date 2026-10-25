# Reproduce the final score study

Run commands from the repository root. The final comparison uses the SAME trained heads for both methods. Baseline uses `--objectness-beta 0`; the proposed score uses `--objectness-beta 1`. There is no second training process for the proposed method.

## 1. Install

The measured head training and final evaluation used Python 3.12.3, PyTorch 2.9.1+cpu, Transformers 4.46.3, NumPy 1.26.4, SciPy 1.13.0, Pillow 10.3.0, and pycocotools 2.0.11 on Windows 11, Intel i5-1340P, 16 GB RAM, four CPU threads. CUDA was not used for these heads or the final confirmation features. Initial pilot features were extracted on a Tesla T4 with PyTorch 2.11.0+cu130 and CUDA 13.0. Recorded environments are in the [training configurations](../results/revision_v2/source_runs/) and [confirmation environment](../results/score_calibration/confirmation_environment.json).

Linux shell (Python 3.12):

```bash
git clone https://github.com/hskad/adaptive-fsvod.git
cd adaptive-fsvod
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-reproduce.txt
```

Windows PowerShell (Python 3.12):

```powershell
git clone https://github.com/hskad/adaptive-fsvod.git
cd adaptive-fsvod
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-reproduce.txt
```

The fixed CPU build has Windows/Linux wheels. macOS users should install its supported PyTorch build instead; cross-platform bitwise equality is not promised. In Colab/Kaggle, retain the installed matched PyTorch/CUDA pair and use `python -m pip install -r requirements.txt`. The [full notebook](../notebooks/score_calibration.ipynb) extracts features and trains from data; it needs a GPU for its encoder cells. A GPU is **not required** for the saved-feature path below.

## 2. Download verified artifacts and reproduce all six evaluations

```bash
python scripts/download_artifacts.py --experiment-dir ../fsvod-reproduction
python scripts/reproduce_results.py --experiment-dir ../fsvod-reproduction --threads 4
```

These commands work in PowerShell too. They download public assets without an account or token, verify archive SHA-256, reject unsafe archive paths, check extracted files, evaluate all three heads twice, and compare AP/AP50/AP75 against the published values. Existing different files are not overwritten. Use a new experiment directory if needed.

- [Shared trained heads, 23.4 MB](https://github.com/hskad/adaptive-fsvod/releases/download/v1.0-results/score_calibration_inference.zip): three inference checkpoints, model configurations, both score presets, source checkpoint hashes, and selected training steps.
- [Measured frozen features, 47.3 MB](https://github.com/hskad/adaptive-fsvod/releases/download/v1.0-results/score_calibration_features.zip): original train/validation features and episodes, plus the exact confirmation features and episodes. No RGB images or encoder weights are included. Tensor storage is compacted without changing values; each tensor was checked after export and reload, and the manifest maps compact file hashes to the measured source byte hashes. Dataset terms still apply.
- [Release and checksum records](https://github.com/hskad/adaptive-fsvod/releases/tag/v1.0-results). Repository copies: [head manifest](../results/score_calibration/inference_bundle.manifest.json), [feature manifest](../results/score_calibration/features_bundle.manifest.json).

Expect these mean scores across seeds 42, 43, and 44:

| Method | AP | AP50 | AP75 |
|---|---:|---:|---:|
| Baseline | 47.439707 | 62.435209 | 50.851907 |
| Objectness score | 58.297585 | 75.043794 | 62.797801 |

Outputs are `../fsvod-reproduction/reproduced/<method>_seed<seed>/{metrics,ground_truth,predictions}.json` and `reproduced/summary.json`. Default matching tolerance is 0.000001 AP points. New hardware or encoder extraction can change rounding; inspect differences rather than silently claiming an exact match. The feature path checks the saved-feature detector and metrics; it does not repeat image extraction.

Expected directory layout after download:

```text
fsvod-reproduction/
  weights/shared_seed{42,43,44}.pt
  fixed_tau060.json
  pilot/{train,val,test}.json
  pilot/{train,val}_episodes.json
  cache/{train,val}/
    metadata.json
    support_pool.json
    frames/<frame-key>.pt
    support/<crop-key>.pt
  score_calibration/confirmation/
    confirmation.json
    episodes.json
    cache/{metadata.json,support_pool.json,frames/,support/}
  reproduced/<method>_seed<seed>/
```

The source manifests contain the original machine's image paths as provenance. Cached evaluation does not read those paths. For raw-image extraction, always supply `--images-root`; do not edit the manifests, because their bytes bind the episodes to the data.

## 3. Reproduce shared-head training

After step 2 supplies the measured train/validation features:

```bash
python scripts/train_fixed_reference.py --experiment-dir ../fsvod-reproduction --threads 4
```

This trains seeds 42/43/44 with the published selected configuration. Each run executes 300 episode updates and validates every 50 updates. The best validation AP selects `best.pt`; `last.pt` contains the full optimizer and RNG state for resume. Outputs are `revision_v2/runs/fixed_tau060_seed<seed>/`, including `config.json`, `training.jsonl`, and validation results. Existing verified runs are kept. A partly completed run resumes from `last.pt`.

The direct executable command for seed 42 is:

```bash
python -m adaptive_fsvod.train --train-manifest ../fsvod-reproduction/pilot/train.json --val-manifest ../fsvod-reproduction/pilot/val.json --train-cache ../fsvod-reproduction/cache/train --val-cache ../fsvod-reproduction/cache/val --train-episodes ../fsvod-reproduction/pilot/train_episodes.json --val-episodes ../fsvod-reproduction/pilot/val_episodes.json --config ../fsvod-reproduction/fixed_tau060.json --output ../fsvod-reproduction/manual_seed42 --steps 300 --eval-every 50 --lr 0.0001 --weight-decay 0.01 --seed 42 --device cpu
```

Use `--resume ../fsvod-reproduction/manual_seed42/last.pt` with the same command to resume. The helper caps CPU threads at four; for direct commands set `OMP_NUM_THREADS=4` and `MKL_NUM_THREADS=4` in your shell if needed. Repeat with seeds 43 and 44 and separate output folders. AdamW uses 5% warmup, cosine decay, gradient norm cap 0.1, classification/box weights 2/5, background weight 0.1, and Hungarian matching with L1 plus generalized IoU. The frozen encoder uses native 960-pixel processing; the head receives 128 candidates of dimension 512. The fixed memory threshold is 0.6 and the cap is 16 tokens per class. The heads have 2,103,815 trainable parameters.

The proposed method uses each resulting `best.pt` unchanged. Evaluate it twice, with beta 0 and beta 1. No proposed optimizer, extra learned tensor, or separate proposed training log exists.

## 4. Evaluate or infer with an individual checkpoint

```bash
python -m adaptive_fsvod.evaluate --manifest ../fsvod-reproduction/score_calibration/confirmation/confirmation.json --cache ../fsvod-reproduction/score_calibration/confirmation/cache --episodes ../fsvod-reproduction/score_calibration/confirmation/episodes.json --checkpoint ../fsvod-reproduction/weights/shared_seed42.pt --objectness-beta 0 --output ../fsvod-reproduction/eval_baseline42 --device cpu
python -m adaptive_fsvod.evaluate --manifest ../fsvod-reproduction/score_calibration/confirmation/confirmation.json --cache ../fsvod-reproduction/score_calibration/confirmation/cache --episodes ../fsvod-reproduction/score_calibration/confirmation/episodes.json --checkpoint ../fsvod-reproduction/weights/shared_seed42.pt --objectness-beta 1 --output ../fsvod-reproduction/eval_objectness42 --device cpu
```

To evaluate a newly trained head, substitute `revision_v2/runs/fixed_tau060_seed42/best.pt` for the downloaded head. Evaluation uses score floor 0.001, class-wise NMS IoU 0.5, and at most 100 boxes per frame. It retains empty frames and resets memory per episode.

For your own ordered video frames and **cropped object examples**, create:

```text
demo/frames/00000.jpg, 00001.jpg, ...
demo/support/cat/example.jpg
demo/support/dog/example.jpg
```

```bash
python -m adaptive_fsvod.infer --frames demo/frames --support demo/support --checkpoint ../fsvod-reproduction/weights/shared_seed42.pt --objectness-beta 1 --output ../fsvod-reproduction/demo_objectness --device cpu
```

The pinned public encoder downloads automatically. Use beta 0 and another output folder for baseline. Outputs: annotated PNG frames, pixel-xyxy boxes and scores in `detections.json`, and frame timings in `timing.json`. The display score threshold is 0.3; use evaluation commands for AP. Raw-image inference uses the historical eager encoder kernel. Confirmation caches use SDPA with the same attention equation and small rounding differences. Cached FPS excludes encoder work; no end-to-end speed gain is claimed.

## 5. Prepare RGB data and rebuild features

Sources, download checks, annotation conversion, original subset selection, and dataset terms are in [DATA.md](DATA.md). The frozen encoder is [google/owlv2-base-patch16-ensemble](https://huggingface.co/google/owlv2-base-patch16-ensemble), revision `cfd3195ba4ea9592eec887ded089f4c08eff231d`. The RGB source is [JiaHuang01/Youtubevos_train](https://huggingface.co/datasets/JiaHuang01/Youtubevos_train), revision `7855adf7e2e8e9386b578f58d490f25c1886368d`. Our downloader fetches selected ZIP ranges, verifies CRC, dimensions and decoding, and records image SHA-256. Do not download the full 9 GB archive.

To reconstruct features for the exact published confirmation selection, after step 2:

```bash
python -m adaptive_fsvod.download_frames --manifests ../fsvod-reproduction/score_calibration/confirmation/confirmation.json --images-root ../fsvod-reproduction/score_calibration/confirmation/images --range-cache ../fsvod-reproduction/download_ranges
python -m adaptive_fsvod.cache --manifest ../fsvod-reproduction/score_calibration/confirmation/confirmation.json --images-root ../fsvod-reproduction/score_calibration/confirmation/images --output ../fsvod-reproduction/rebuilt_confirmation_cache --revision cfd3195ba4ea9592eec887ded089f4c08eff231d --candidates 128 --support-per-class 4 --attention-backend sdpa --device cpu
```

Use `--device cuda` if available. Evaluate against `rebuilt_confirmation_cache` by changing `--cache`. This uses a separate directory and preserves the measured features. For training caches, run the downloader with `pilot/train.json` and `pilot/val.json`, then cache each manifest into a new directory with `--support-per-class 16 --attention-backend eager` and the same encoder revision. The [full notebook](../notebooks/score_calibration.ipynb) includes annotation conversion, original subset/episode generation, all three shared-head training runs, bounded validation search, and the frozen fresh confirmation. Historical excluded class/video IDs are supplied by published manifests; failed methods need not be trained again.

Preprocessing normalizes pixel boxes to the native padded square, keeps encoder features fixed, and stores query features as float16 and support vectors as float32. Both compared scores use the same feature tensors. Source labels and RGB images are not part of the learned head weights. Images are not hosted by this repository; the derived features retain the dataset's research-use terms.

## 6. Evidence, uncertainty, and software checks

- [Final results, class effects, protocols, and raw detections](../results/score_calibration/README.md).
- Training logs/configurations: [seed 42](../results/revision_v2/source_runs/fixed_tau060_seed42/), [43](../results/revision_v2/source_runs/fixed_tau060_seed43/), [44](../results/revision_v2/source_runs/fixed_tau060_seed44/).
- [Predeclared score study](SCORE_CALIBRATION_PLAN.md); beta was chosen only on validation from 0/0.5/1. Its selected value is 1. The six-class confirmation was fixed before scoring.
- `scripts/analyse_score_confirmation.py` contains the paired video-bootstrap analysis. All 1,000 draw multiplicities and AP values are already saved in `results/score_calibration/bootstrap_draws.json`; the conditional 95% interval is [+5.960, +14.850] AP points. It does not cover new classes, support selection, or the full model-development process. Bear gives the largest gain.

Recompute all six AP sets, class/video diagnostics, and the 1,000-draw interval directly from the public saved detections, without model weights or original machine paths:

```bash
python scripts/analyse_score_confirmation.py --published-results results/score_calibration --output ../fsvod-reproduction/uncertainty
```

This verifies the source file hashes and recomputes the statistics; it does not run the detector. Use section 2 to reproduce detector outputs. The full cloud notebook should start in a new experiment directory, separate from downloaded measured bundles.

```bash
python -m pytest -q
python -m adaptive_fsvod.smoke --output ../fsvod-reproduction/synthetic_check --steps 4
```

The synthetic check verifies software only. The result is a custom small FSYTV subset, not an exact reproduction of the paper's full benchmarks. Earlier failed experiments remain in `results/`. See [VALIDATION.md](VALIDATION.md) for tested paths and their limits.
